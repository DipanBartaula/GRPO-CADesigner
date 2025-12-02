import torch
import numpy as np
from typing import Dict, List, Optional
import trimesh
from utils import cad_code_to_mesh, mesh_to_point_cloud, render_mesh, check_code_syntax, normalize_point_cloud
import subprocess
import tempfile
import os

class CodeCompilationReward:
    """Reward for code compilation success"""
    def __init__(self, success_reward: float = 1.0, failure_penalty: float = -0.5):
        self.success_reward = success_reward
        self.failure_penalty = failure_penalty
    
    def compute(self, code: str) -> float:
        """Check if code compiles"""
        if check_code_syntax(code):
            return self.success_reward
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
        try:
            mesh = cad_code_to_mesh(code)
            if mesh is not None and len(mesh.vertices) > 0:
                return self.success_reward, mesh
            else:
                return self.runtime_error_penalty, None
        except Exception as e:
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
    """Main reward computer combining all reward functions"""
    def __init__(
        self,
        reward_models,
        device: torch.device,
        compilation_weight: float = 0.1,
        execution_weight: float = 0.15,
        cad_specific_weight: float = 0.15,
        reward_model_weight: float = 0.6
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
    
    def compute_rewards(
        self,
        generated_codes: List[str],
        prompts: Optional[List[str]] = None
    ) -> tuple[torch.Tensor, Dict[str, List[float]]]:
        """
        Compute all rewards for generated CAD codes
        
        Returns:
            total_rewards: (batch_size,) tensor
            detailed_rewards: Dictionary of individual reward components
        """
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
            # 1. Compilation reward
            comp_reward = self.compilation_reward.compute(code)
            detailed_rewards['compilation'].append(comp_reward)
            
            # 2. Execution reward
            exec_reward, mesh = self.execution_reward.compute(code)
            detailed_rewards['execution'].append(exec_reward)
            
            # 3. CAD-specific rewards
            if mesh is not None:
                cad_rewards = self.cad_reward.compute(mesh)
                cad_total = self.cad_reward.compute_weighted_reward(cad_rewards)
            else:
                cad_total = 0.0
            detailed_rewards['cad_specific'].append(cad_total)
            
            # 4. Reward model scores
            rm_rewards = self._compute_reward_model_scores(mesh, code)
            for key, value in rm_rewards.items():
                detailed_rewards[key].append(value)
            
            # Compute total weighted reward
            total = (
                self.weights['compilation'] * comp_reward +
                self.weights['execution'] * exec_reward +
                self.weights['cad_specific'] * cad_total +
                self.weights['reward_models'] * sum(rm_rewards.values()) / max(len(rm_rewards), 1)
            )
            
            total_rewards[i] = total
        
        return total_rewards, detailed_rewards
    
    def _compute_reward_model_scores(
        self,
        mesh: Optional[trimesh.Trimesh],
        code: str
    ) -> Dict[str, float]:
        """Compute scores from all reward models"""
        scores = {}
        
        if mesh is None:
            return {name: 0.0 for name in ['pointbert', 'ulip2', 'multiview_clip', 'pointclip', 'geometric']}
        
        try:
            # Prepare inputs for reward models
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