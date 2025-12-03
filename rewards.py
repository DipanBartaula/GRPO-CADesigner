import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import trimesh
from utils import cad_code_to_mesh, mesh_to_point_cloud, render_mesh, check_code_syntax, normalize_point_cloud
import subprocess
import tempfile
import os
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import traceback
import time


# ============================================================================
# MULTIPROCESSING WORKER FUNCTIONS (must be at module level for pickling)
# ============================================================================

def _worker_check_compilation(args: Tuple[int, str, float, float]) -> Tuple[int, float, str]:
    """
    Worker function to check code compilation in a separate process.
    
    Args:
        args: (index, code, success_reward, failure_penalty)
    
    Returns:
        (index, reward, debug_message)
    """
    idx, code, success_reward, failure_penalty = args
    
    if not code or not code.strip():
        return idx, failure_penalty, "EMPTY CODE"
    
    try:
        compile(code, '<string>', 'exec')
        return idx, success_reward, "SYNTAX OK"
    except SyntaxError as e:
        msg = f"SYNTAX ERROR at line {e.lineno}: {e.msg}"
        return idx, failure_penalty, msg
    except Exception as e:
        msg = f"ERROR: {type(e).__name__}: {e}"
        return idx, failure_penalty, msg


def _worker_execute_and_mesh(args: Tuple[int, str, float, float, int]) -> Tuple[int, float, Optional[bytes], str]:
    """
    Worker function to execute code and convert to mesh in a separate process.
    
    Args:
        args: (index, code, success_reward, failure_penalty, timeout_seconds)
    
    Returns:
        (index, reward, serialized_mesh_bytes or None, debug_message)
    """
    idx, code, success_reward, failure_penalty, timeout_seconds = args
    
    if not code or not code.strip():
        return idx, failure_penalty, None, "EMPTY CODE"
    
    try:
        mesh = cad_code_to_mesh(code)
        if mesh is not None and len(mesh.vertices) > 0:
            # Serialize mesh to bytes for transfer between processes
            mesh_bytes = mesh.export(file_type='stl')
            msg = f"SUCCESS - {len(mesh.vertices)} vertices, {len(mesh.faces)} faces"
            return idx, success_reward, mesh_bytes, msg
        else:
            return idx, failure_penalty, None, "FAILED - no valid mesh"
    except Exception as e:
        msg = f"EXCEPTION: {type(e).__name__}: {str(e)[:100]}"
        return idx, failure_penalty, None, msg


def _worker_mesh_to_features(args: Tuple[int, bytes, int]) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Worker function to convert mesh to point cloud and compute CAD features.
    
    Args:
        args: (index, mesh_bytes, num_points)
    
    Returns:
        (index, point_cloud or None, cad_rewards_array or None, debug_message)
    """
    idx, mesh_bytes, num_points = args
    
    if mesh_bytes is None:
        return idx, None, None, "NO MESH"
    
    try:
        # Deserialize mesh from bytes
        mesh = trimesh.load(trimesh.util.wrap_as_stream(mesh_bytes), file_type='stl')
        
        # Convert to point cloud
        point_cloud = mesh_to_point_cloud(mesh, num_points=num_points)
        point_cloud = normalize_point_cloud(point_cloud)
        
        # Compute CAD-specific rewards
        cad_rewards = {}
        
        # Vertex count
        num_vertices = len(mesh.vertices)
        cad_rewards['vertex_count'] = _score_count(num_vertices, optimal=1000, min_val=50, max_val=5000)
        
        # Face count
        num_faces = len(mesh.faces)
        cad_rewards['face_count'] = _score_count(num_faces, optimal=2000, min_val=100, max_val=10000)
        
        # Watertight
        cad_rewards['watertight'] = 1.0 if mesh.is_watertight else 0.0
        
        # Manifold
        try:
            cad_rewards['manifold'] = 1.0 if mesh.is_winding_consistent else 0.5
        except:
            cad_rewards['manifold'] = 0.5
        
        # Valid topology
        cad_rewards['valid_topology'] = 1.0 if mesh.is_valid else 0.0
        
        # Complexity
        euler = mesh.euler_number
        cad_rewards['complexity'] = _score_complexity(euler)
        
        # Convert to array for transfer
        cad_rewards_arr = np.array([
            cad_rewards['vertex_count'],
            cad_rewards['face_count'],
            cad_rewards['watertight'],
            cad_rewards['manifold'],
            cad_rewards['valid_topology'],
            cad_rewards['complexity']
        ])
        
        msg = f"SUCCESS - {num_points} points, CAD features computed"
        return idx, point_cloud, cad_rewards_arr, msg
        
    except Exception as e:
        msg = f"EXCEPTION: {type(e).__name__}: {str(e)[:100]}"
        return idx, None, None, msg


def _score_count(count: int, optimal: int, min_val: int, max_val: int) -> float:
    """Score a count value with optimal target"""
    if count < min_val:
        return count / min_val * 0.5
    elif count > max_val:
        return max(0.0, 1.0 - (count - max_val) / max_val)
    else:
        diff = abs(count - optimal) / optimal
        return np.exp(-diff**2)


def _score_complexity(euler: int) -> float:
    """Score topological complexity"""
    if euler == 2:
        return 0.8
    elif abs(euler) <= 10:
        return 1.0
    else:
        return 0.6

class CodeCompilationReward:
    """Reward for code compilation success"""
    def __init__(self, success_reward: float = 1.0, failure_penalty: float = -0.5):
        self.success_reward = success_reward
        self.failure_penalty = failure_penalty
    
    def compute(self, code: str) -> float:
        """Check if code compiles"""
        # Debug: Check code status
        code_len = len(code) if code else 0
        is_empty = not code or not code.strip()
        
        if is_empty:
            print(f"      [Compilation] EMPTY CODE - returning penalty {self.failure_penalty}")
            return self.failure_penalty
        
        syntax_ok = check_code_syntax(code)
        if syntax_ok:
            print(f"      [Compilation] SYNTAX OK - returning reward {self.success_reward}")
            return self.success_reward
        else:
            # Try to get more info about the syntax error
            try:
                compile(code, '<string>', 'exec')
            except SyntaxError as e:
                print(f"      [Compilation] SYNTAX ERROR at line {e.lineno}: {e.msg}")
                print(f"        Problematic line: {e.text}")
            except Exception as e:
                print(f"      [Compilation] OTHER ERROR: {type(e).__name__}: {e}")
            print(f"      [Compilation] FAILED - returning penalty {self.failure_penalty}")
            return self.failure_penalty


class CodeExecutionReward:
    """Reward for successful code execution"""
    def __init__(
        self,
        success_reward: float = 1.0,
        runtime_error_penalty: float = -0.5,
        timeout_penalty: float = -0.3,
        timeout_seconds: int = 5
    ):
        self.success_reward = success_reward
        self.runtime_error_penalty = runtime_error_penalty
        self.timeout_penalty = timeout_penalty
        self.timeout_seconds = timeout_seconds
    
    def compute(self, code: str) -> tuple[float, Optional[trimesh.Trimesh]]:
        """Execute code and return reward + mesh if successful"""
        if not code or not code.strip():
            print(f"      [Execution] EMPTY CODE - returning penalty {self.runtime_error_penalty}")
            return self.runtime_error_penalty, None
        
        try:
            print(f"      [Execution] Attempting to execute code ({len(code)} chars)...")
            mesh = cad_code_to_mesh(code)
            if mesh is not None and len(mesh.vertices) > 0:
                print(f"      [Execution] SUCCESS - mesh has {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                return self.success_reward, mesh
            else:
                if mesh is None:
                    print(f"      [Execution] FAILED - cad_code_to_mesh returned None")
                else:
                    print(f"      [Execution] FAILED - mesh has 0 vertices")
                return self.runtime_error_penalty, None
        except Exception as e:
            print(f"      [Execution] EXCEPTION: {type(e).__name__}: {e}")
            return self.runtime_error_penalty, None


class CADSpecificReward:
    """CAD-specific rewards for quality metrics"""
    def __init__(self):
        self.weights = {
            'vertex_count': 0.1,
            'face_count': 0.1,
            'watertight': 0.3,
            'manifold': 0.2,
            'valid_topology': 0.2,
            'complexity': 0.1
        }
    
    def compute(self, mesh: trimesh.Trimesh) -> Dict[str, float]:
        """Compute CAD-specific rewards"""
        if mesh is None:
            return {key: 0.0 for key in self.weights.keys()}
        
        rewards = {}
        
        try:
            # Vertex count (normalized, prefer moderate complexity)
            num_vertices = len(mesh.vertices)
            rewards['vertex_count'] = self._score_count(num_vertices, optimal=1000, min_val=50, max_val=5000)
            
            # Face count (normalized)
            num_faces = len(mesh.faces)
            rewards['face_count'] = self._score_count(num_faces, optimal=2000, min_val=100, max_val=10000)
            
            # Watertight (closed mesh)
            rewards['watertight'] = 1.0 if mesh.is_watertight else 0.0
            
            # Manifold (valid edges)
            try:
                rewards['manifold'] = 1.0 if mesh.is_winding_consistent else 0.5
            except:
                rewards['manifold'] = 0.5
            
            # Valid topology
            rewards['valid_topology'] = 1.0 if mesh.is_valid else 0.0
            
            # Complexity (euler number, convexity)
            euler = mesh.euler_number
            rewards['complexity'] = self._score_complexity(euler)
            
        except Exception as e:
            print(f"Error computing CAD rewards: {e}")
            rewards = {key: 0.0 for key in self.weights.keys()}
        
        return rewards
    
    def _score_count(self, count: int, optimal: int, min_val: int, max_val: int) -> float:
        """Score a count value with optimal target"""
        if count < min_val:
            return count / min_val * 0.5
        elif count > max_val:
            return max(0.0, 1.0 - (count - max_val) / max_val)
        else:
            # Gaussian around optimal
            diff = abs(count - optimal) / optimal
            return np.exp(-diff**2)
    
    def _score_complexity(self, euler: int) -> float:
        """Score topological complexity"""
        # Euler characteristic of 2 indicates a sphere (simple)
        # Other values indicate more complex topology
        if euler == 2:
            return 0.8  # Simple but valid
        elif abs(euler) <= 10:
            return 1.0  # Moderately complex
        else:
            return 0.6  # Very complex
    
    def compute_weighted_reward(self, rewards: Dict[str, float]) -> float:
        """Compute weighted total reward"""
        total = sum(self.weights[key] * rewards[key] for key in rewards)
        return total


class RewardComputer:
    """Main reward computer combining all reward functions with multiprocessing support"""
    def __init__(
        self,
        reward_models,
        device: torch.device,
        compilation_weight: float = 0.1,
        execution_weight: float = 0.15,
        cad_specific_weight: float = 0.15,
        reward_model_weight: float = 0.6,
        num_workers: int = None,
        use_multiprocessing: bool = True
    ):
        self.reward_models = reward_models
        self.device = device
        
        self.compilation_reward = CodeCompilationReward()
        self.execution_reward = CodeExecutionReward()
        self.cad_reward = CADSpecificReward()
        
        self.weights = {
            'compilation': compilation_weight,
            'execution': execution_weight,
            'cad_specific': cad_specific_weight,
            'reward_models': reward_model_weight
        }
        
        # Multiprocessing settings
        self.use_multiprocessing = use_multiprocessing
        self.num_workers = num_workers if num_workers else max(1, cpu_count() - 1)
        print(f"[RewardComputer] Initialized with {self.num_workers} workers, multiprocessing={use_multiprocessing}")
    
    def compute_rewards(
        self,
        generated_codes: List[str],
        prompts: Optional[List[str]] = None
    ) -> tuple[torch.Tensor, Dict[str, List[float]]]:
        """
        Compute all rewards for generated CAD codes using multiprocessing.
        
        Returns:
            total_rewards: (batch_size,) tensor
            detailed_rewards: Dictionary of individual reward components
        """
        batch_size = len(generated_codes)
        
        print(f"\n{'='*60}")
        print(f"REWARD COMPUTATION DEBUG INFO (PARALLEL)")
        print(f"{'='*60}")
        print(f"Processing {batch_size} scripts for reward computation")
        print(f"Using multiprocessing: {self.use_multiprocessing}, Workers: {self.num_workers}")
        print(f"Reward weights: {self.weights}")
        
        if self.use_multiprocessing and batch_size > 1:
            return self._compute_rewards_parallel(generated_codes, prompts)
        else:
            return self._compute_rewards_sequential(generated_codes, prompts)
    
    def _compute_rewards_parallel(
        self,
        generated_codes: List[str],
        prompts: Optional[List[str]] = None
    ) -> tuple[torch.Tensor, Dict[str, List[float]]]:
        """Compute rewards using multiprocessing for parallelization."""
        batch_size = len(generated_codes)
        
        detailed_rewards = {
            'compilation': [0.0] * batch_size,
            'execution': [0.0] * batch_size,
            'cad_specific': [0.0] * batch_size,
            'pointbert': [0.0] * batch_size,
            'ulip2': [0.0] * batch_size,
            'multiview_clip': [0.0] * batch_size,
            'pointclip': [0.0] * batch_size,
            'geometric': [0.0] * batch_size
        }
        
        total_rewards = torch.zeros(batch_size, device=self.device)
        meshes = [None] * batch_size
        mesh_bytes_list = [None] * batch_size
        point_clouds = [None] * batch_size
        cad_rewards_list = [None] * batch_size
        
        # ============================================================
        # PHASE 1: Parallel Compilation Check
        # ============================================================
        print(f"\n  [PHASE 1] Parallel Compilation Check ({batch_size} scripts)...")
        phase1_start = time.time()
        
        comp_args = [
            (i, code, self.compilation_reward.success_reward, self.compilation_reward.failure_penalty)
            for i, code in enumerate(generated_codes)
        ]
        
        try:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(_worker_check_compilation, arg): arg[0] for arg in comp_args}
                for future in as_completed(futures):
                    try:
                        idx, reward, msg = future.result(timeout=10)
                        detailed_rewards['compilation'][idx] = reward
                        status = "PASSED" if reward > 0 else "FAILED"
                        print(f"    Script [{idx}] Compilation: {status} - {msg}")
                    except Exception as e:
                        idx = futures[future]
                        detailed_rewards['compilation'][idx] = self.compilation_reward.failure_penalty
                        print(f"    Script [{idx}] Compilation: FAILED - Worker error: {e}")
        except Exception as e:
            print(f"    ⚠️ ProcessPoolExecutor failed: {e}, falling back to sequential")
            for i, code in enumerate(generated_codes):
                detailed_rewards['compilation'][i] = self.compilation_reward.compute(code)
        
        phase1_time = time.time() - phase1_start
        print(f"  [PHASE 1] Completed in {phase1_time:.2f}s")
        
        # ============================================================
        # PHASE 2: Parallel Execution & Mesh Generation
        # ============================================================
        print(f"\n  [PHASE 2] Parallel Execution & Mesh Generation ({batch_size} scripts)...")
        phase2_start = time.time()
        
        exec_args = [
            (i, code, self.execution_reward.success_reward, self.execution_reward.runtime_error_penalty, 
             self.execution_reward.timeout_seconds)
            for i, code in enumerate(generated_codes)
        ]
        
        try:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(_worker_execute_and_mesh, arg): arg[0] for arg in exec_args}
                for future in as_completed(futures):
                    try:
                        idx, reward, mesh_bytes, msg = future.result(timeout=30)
                        detailed_rewards['execution'][idx] = reward
                        mesh_bytes_list[idx] = mesh_bytes
                        status = "PASSED" if reward > 0 else "FAILED"
                        print(f"    Script [{idx}] Execution: {status} - {msg}")
                    except Exception as e:
                        idx = futures[future]
                        detailed_rewards['execution'][idx] = self.execution_reward.runtime_error_penalty
                        print(f"    Script [{idx}] Execution: FAILED - Worker error: {e}")
        except Exception as e:
            print(f"    ⚠️ ProcessPoolExecutor failed: {e}, falling back to sequential")
            for i, code in enumerate(generated_codes):
                reward, mesh = self.execution_reward.compute(code)
                detailed_rewards['execution'][i] = reward
                if mesh is not None:
                    mesh_bytes_list[i] = mesh.export(file_type='stl')
        
        phase2_time = time.time() - phase2_start
        print(f"  [PHASE 2] Completed in {phase2_time:.2f}s")
        
        # ============================================================
        # PHASE 3: Parallel Point Cloud & CAD Feature Extraction
        # ============================================================
        # Only process scripts that have valid meshes
        valid_mesh_indices = [i for i, mb in enumerate(mesh_bytes_list) if mb is not None]
        print(f"\n  [PHASE 3] Parallel Point Cloud & CAD Features ({len(valid_mesh_indices)} valid meshes)...")
        phase3_start = time.time()
        
        if valid_mesh_indices:
            feature_args = [
                (i, mesh_bytes_list[i], 2048)
                for i in valid_mesh_indices
            ]
            
            try:
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    futures = {executor.submit(_worker_mesh_to_features, arg): arg[0] for arg in feature_args}
                    for future in as_completed(futures):
                        try:
                            idx, pc, cad_arr, msg = future.result(timeout=30)
                            point_clouds[idx] = pc
                            cad_rewards_list[idx] = cad_arr
                            print(f"    Script [{idx}] Features: {msg}")
                        except Exception as e:
                            idx = futures[future]
                            print(f"    Script [{idx}] Features: FAILED - Worker error: {e}")
            except Exception as e:
                print(f"    ⚠️ ProcessPoolExecutor failed: {e}, falling back to sequential")
                for i in valid_mesh_indices:
                    try:
                        mesh = trimesh.load(trimesh.util.wrap_as_stream(mesh_bytes_list[i]), file_type='stl')
                        meshes[i] = mesh
                        point_clouds[i] = normalize_point_cloud(mesh_to_point_cloud(mesh, num_points=2048))
                        cad_rewards = self.cad_reward.compute(mesh)
                        cad_rewards_list[i] = np.array([
                            cad_rewards['vertex_count'], cad_rewards['face_count'],
                            cad_rewards['watertight'], cad_rewards['manifold'],
                            cad_rewards['valid_topology'], cad_rewards['complexity']
                        ])
                    except Exception as inner_e:
                        print(f"    Script [{i}] Features: FAILED - {inner_e}")
        
        phase3_time = time.time() - phase3_start
        print(f"  [PHASE 3] Completed in {phase3_time:.2f}s")
        
        # ============================================================
        # PHASE 4: Compute CAD-specific rewards and Reward Model Scores
        # ============================================================
        print(f"\n  [PHASE 4] Computing CAD & Reward Model Scores...")
        phase4_start = time.time()
        
        cad_weights = np.array([0.1, 0.1, 0.3, 0.2, 0.2, 0.1])  # vertex, face, watertight, manifold, valid, complexity
        
        for i in range(batch_size):
            # CAD-specific reward
            if cad_rewards_list[i] is not None:
                cad_total = float(np.dot(cad_weights, cad_rewards_list[i]))
                detailed_rewards['cad_specific'][i] = cad_total
                print(f"    Script [{i}] CAD: reward={cad_total:.3f}")
            else:
                detailed_rewards['cad_specific'][i] = 0.0
                print(f"    Script [{i}] CAD: SKIPPED (no mesh)")
            
            # Reward model scores (needs mesh deserialized for GPU models)
            if mesh_bytes_list[i] is not None and point_clouds[i] is not None:
                try:
                    mesh = trimesh.load(trimesh.util.wrap_as_stream(mesh_bytes_list[i]), file_type='stl')
                    rm_rewards = self._compute_reward_model_scores(mesh, generated_codes[i], point_clouds[i])
                    for key, value in rm_rewards.items():
                        detailed_rewards[key][i] = value
                    print(f"    Script [{i}] Reward Models: {rm_rewards}")
                except Exception as e:
                    print(f"    Script [{i}] Reward Models: FAILED - {e}")
            else:
                print(f"    Script [{i}] Reward Models: SKIPPED (no mesh)")
        
        phase4_time = time.time() - phase4_start
        print(f"  [PHASE 4] Completed in {phase4_time:.2f}s")
        
        # ============================================================
        # PHASE 5: Compute Total Rewards
        # ============================================================
        print(f"\n  [PHASE 5] Computing Total Weighted Rewards...")
        
        for i in range(batch_size):
            comp_reward = detailed_rewards['compilation'][i]
            exec_reward = detailed_rewards['execution'][i]
            cad_total = detailed_rewards['cad_specific'][i]
            
            rm_values = [detailed_rewards[key][i] for key in ['pointbert', 'ulip2', 'multiview_clip', 'pointclip', 'geometric']]
            rm_total = sum(rm_values) / max(len(rm_values), 1)
            
            total = (
                self.weights['compilation'] * comp_reward +
                self.weights['execution'] * exec_reward +
                self.weights['cad_specific'] * cad_total +
                self.weights['reward_models'] * rm_total
            )
            
            total_rewards[i] = total
            print(f"    Script [{i}] TOTAL: {total:.4f} (comp={comp_reward:.2f}, exec={exec_reward:.2f}, cad={cad_total:.2f}, rm={rm_total:.2f})")
        
        # Summary
        self._print_summary(total_rewards, detailed_rewards, batch_size)
        
        total_time = phase1_time + phase2_time + phase3_time + phase4_time
        print(f"\n  TOTAL PARALLEL PROCESSING TIME: {total_time:.2f}s")
        
        return total_rewards, detailed_rewards
    
    def _compute_rewards_sequential(
        self,
        generated_codes: List[str],
        prompts: Optional[List[str]] = None
    ) -> tuple[torch.Tensor, Dict[str, List[float]]]:
        """Original sequential reward computation (fallback)."""
        batch_size = len(generated_codes)
        detailed_rewards = {
            'compilation': [],
            'execution': [],
            'cad_specific': [],
            'pointbert': [],
            'ulip2': [],
            'multiview_clip': [],
            'pointclip': [],
            'geometric': []
        }
        
        total_rewards = torch.zeros(batch_size, device=self.device)
        
        for i, code in enumerate(generated_codes):
            print(f"\n  --- Script [{i}] Reward Breakdown (Sequential) ---")
            code_len = len(code) if code else 0
            is_empty = not code or not code.strip()
            print(f"  Code length: {code_len} chars, Is empty: {is_empty}")
            
            # 1. Compilation reward
            comp_reward = self.compilation_reward.compute(code)
            detailed_rewards['compilation'].append(comp_reward)
            comp_status = "PASSED" if comp_reward > 0 else "FAILED"
            print(f"  [1] Compilation: {comp_status} (reward={comp_reward:.3f})")
            
            # 2. Execution reward
            exec_reward, mesh = self.execution_reward.compute(code)
            detailed_rewards['execution'].append(exec_reward)
            exec_status = "PASSED" if exec_reward > 0 else "FAILED"
            mesh_status = f"mesh with {len(mesh.vertices)} vertices" if mesh is not None else "NO MESH"
            print(f"  [2] Execution: {exec_status} (reward={exec_reward:.3f}, {mesh_status})")
            
            # 3. CAD-specific rewards
            if mesh is not None:
                cad_rewards = self.cad_reward.compute(mesh)
                cad_total = self.cad_reward.compute_weighted_reward(cad_rewards)
                print(f"  [3] CAD-specific: reward={cad_total:.3f}")
                print(f"      Details: {cad_rewards}")
            else:
                cad_total = 0.0
                print(f"  [3] CAD-specific: SKIPPED (no mesh) reward=0.0")
            detailed_rewards['cad_specific'].append(cad_total)
            
            # 4. Reward model scores
            rm_rewards = self._compute_reward_model_scores_legacy(mesh, code)
            for key, value in rm_rewards.items():
                detailed_rewards[key].append(value)
            rm_total = sum(rm_rewards.values()) / max(len(rm_rewards), 1)
            print(f"  [4] Reward models: avg={rm_total:.3f}")
            print(f"      Details: {rm_rewards}")
            
            # Compute total weighted reward
            total = (
                self.weights['compilation'] * comp_reward +
                self.weights['execution'] * exec_reward +
                self.weights['cad_specific'] * cad_total +
                self.weights['reward_models'] * rm_total
            )
            
            print(f"  [TOTAL] Weighted reward for script [{i}]: {total:.4f}")
            total_rewards[i] = total
        
        self._print_summary(total_rewards, detailed_rewards, batch_size)
        return total_rewards, detailed_rewards
    
    def _print_summary(self, total_rewards, detailed_rewards, batch_size):
        """Print reward computation summary."""
        print(f"\n{'='*60}")
        print(f"REWARD COMPUTATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total rewards tensor: {total_rewards}")
        print(f"Mean reward: {total_rewards.mean():.4f}")
        print(f"Std reward: {total_rewards.std():.4f}")
        print(f"Min reward: {total_rewards.min():.4f}")
        print(f"Max reward: {total_rewards.max():.4f}")
        all_same = total_rewards.std() < 1e-6
        if all_same:
            print(f"⚠️ WARNING: All rewards are nearly identical! This will produce zero gradients.")
        num_comp_pass = sum(1 for r in detailed_rewards['compilation'] if r > 0)
        num_exec_pass = sum(1 for r in detailed_rewards['execution'] if r > 0)
        print(f"Compilation passed: {num_comp_pass}/{batch_size}")
        print(f"Execution passed: {num_exec_pass}/{batch_size}")
        print(f"{'='*60}\n")
    
    def _compute_reward_model_scores(
        self,
        mesh: Optional[trimesh.Trimesh],
        code: str,
        point_cloud: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute scores from all reward models (with pre-computed point cloud)."""
        scores = {}
        
        if mesh is None:
            return {name: 0.0 for name in ['pointbert', 'ulip2', 'multiview_clip', 'pointclip', 'geometric']}
        
        try:
            # Use pre-computed point cloud or compute new one
            if point_cloud is None:
                point_cloud = mesh_to_point_cloud(mesh, num_points=2048)
                point_cloud = normalize_point_cloud(point_cloud)
            
            point_cloud_tensor = torch.tensor(point_cloud, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # Rendered views
            rendered_views = render_mesh(mesh, views=4)
            rendered_views_tensor = torch.tensor(
                np.stack(rendered_views),
                dtype=torch.float32,
                device=self.device
            ).permute(0, 3, 1, 2).unsqueeze(0) / 255.0
            
            # Geometric features
            geometric_features = self.reward_models.models['geometric'].extract_geometric_features(mesh)
            geometric_features = geometric_features.unsqueeze(0).to(self.device)
            
            # Prepare inputs
            inputs = {
                'point_cloud': point_cloud_tensor,
                'rendered_views': rendered_views_tensor,
                'geometric_features': geometric_features
            }
            
            # Compute rewards
            with torch.no_grad():
                model_rewards = self.reward_models(inputs)
            
            # Convert to float scores
            for name, reward_tensor in model_rewards.items():
                scores[name] = reward_tensor.mean().item()
        
        except Exception as e:
            print(f"Error computing reward model scores: {e}")
            scores = {name: 0.0 for name in ['pointbert', 'ulip2', 'multiview_clip', 'pointclip', 'geometric']}
        
        return scores
    
    def _compute_reward_model_scores_legacy(
        self,
        mesh: Optional[trimesh.Trimesh],
        code: str
    ) -> Dict[str, float]:
        """Legacy method for sequential processing."""
        return self._compute_reward_model_scores(mesh, code, None)
    
    def normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize rewards to have mean 0 and std 1"""
        if len(rewards) > 1:
            mean = rewards.mean()
            std = rewards.std() + 1e-8
            return (rewards - mean) / std
        return rewards


def compute_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GAE (Generalized Advantage Estimation)
    
    Args:
        rewards: (batch_size,)
        values: (batch_size,)
        gamma: discount factor
        lam: GAE lambda parameter
    
    Returns:
        advantages: (batch_size,)
        returns: (batch_size,)
    """
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        returns[t] = gae + values[t]
    
    return advantages, returns