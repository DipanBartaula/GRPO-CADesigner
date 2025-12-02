"""
Test script for mesh conversion, point cloud conversion, and rendering
Tests all the CAD processing pipeline components
"""

import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path
import traceback

from utils import (
    cad_code_to_mesh,
    mesh_to_point_cloud,
    render_mesh,
    normalize_point_cloud,
    log_rendered_mesh_to_wandb
)

class MeshConversionTester:
    """Test mesh conversion pipeline"""
    
    def __init__(self, output_dir: str = "test_outputs/mesh_conversion"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def test_code_to_mesh(self, code: str, name: str) -> bool:
        """Test CAD code to mesh conversion"""
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")
        
        try:
            # Convert code to mesh
            print("Converting code to mesh...")
            mesh = cad_code_to_mesh(code)
            
            if mesh is None:
                print("✗ Failed: Mesh is None")
                return False
            
            print(f"✓ Mesh created successfully")
            print(f"  - Vertices: {len(mesh.vertices)}")
            print(f"  - Faces: {len(mesh.faces)}")
            print(f"  - Watertight: {mesh.is_watertight}")
            print(f"  - Volume: {mesh.volume if mesh.is_watertight else 'N/A'}")
            print(f"  - Surface Area: {mesh.area:.4f}")
            
            # Save mesh
            mesh_path = os.path.join(self.output_dir, f"{name}_mesh.obj")
            mesh.export(mesh_path)
            print(f"✓ Mesh saved to: {mesh_path}")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed with error: {e}")
            traceback.print_exc()
            return False
    
    def test_mesh_to_pointcloud(self, code: str, name: str, num_points: int = 2048) -> bool:
        """Test mesh to point cloud conversion"""
        print(f"\nTesting point cloud conversion for: {name}")
        
        try:
            # Get mesh
            mesh = cad_code_to_mesh(code)
            if mesh is None:
                print("✗ Cannot test point cloud: mesh is None")
                return False
            
            # Convert to point cloud
            print(f"Converting to point cloud ({num_points} points)...")
            point_cloud = mesh_to_point_cloud(mesh, num_points=num_points)
            
            print(f"✓ Point cloud created")
            print(f"  - Shape: {point_cloud.shape}")
            print(f"  - Min coords: {point_cloud.min(axis=0)}")
            print(f"  - Max coords: {point_cloud.max(axis=0)}")
            print(f"  - Mean coords: {point_cloud.mean(axis=0)}")
            
            # Normalize point cloud
            print("Normalizing point cloud...")
            normalized_pc = normalize_point_cloud(point_cloud)
            
            print(f"✓ Point cloud normalized")
            print(f"  - Min coords: {normalized_pc.min(axis=0)}")
            print(f"  - Max coords: {normalized_pc.max(axis=0)}")
            print(f"  - Mean coords: {normalized_pc.mean(axis=0)}")
            
            # Visualize point cloud
            fig = plt.figure(figsize=(12, 5))
            
            # Original
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], 
                       c=point_cloud[:, 2], cmap='viridis', s=1)
            ax1.set_title(f'{name} - Original Point Cloud')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            
            # Normalized
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.scatter(normalized_pc[:, 0], normalized_pc[:, 1], normalized_pc[:, 2],
                       c=normalized_pc[:, 2], cmap='viridis', s=1)
            ax2.set_title(f'{name} - Normalized Point Cloud')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            
            plt.tight_layout()
            pc_path = os.path.join(self.output_dir, f"{name}_pointcloud.png")
            plt.savefig(pc_path, dpi=150)
            plt.close()
            
            print(f"✓ Point cloud visualization saved to: {pc_path}")
            
            return True
            
        except Exception as e:
            print(f"✗ Point cloud conversion failed: {e}")
            traceback.print_exc()
            return False
    
    def test_mesh_rendering(self, code: str, name: str, num_views: int = 4) -> bool:
        """Test mesh rendering"""
        print(f"\nTesting rendering for: {name}")
        
        try:
            # Get mesh
            mesh = cad_code_to_mesh(code)
            if mesh is None:
                print("✗ Cannot test rendering: mesh is None")
                return False
            
            # Render mesh
            print(f"Rendering {num_views} views...")
            rendered_views = render_mesh(mesh, views=num_views)
            
            print(f"✓ Rendering completed")
            print(f"  - Number of views: {len(rendered_views)}")
            print(f"  - View shape: {rendered_views[0].shape}")
            
            # Visualize all views
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            
            for idx, (ax, view) in enumerate(zip(axes.flat, rendered_views)):
                ax.imshow(view)
                ax.axis('off')
                ax.set_title(f'{name} - View {idx+1}', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            render_path = os.path.join(self.output_dir, f"{name}_rendered.png")
            plt.savefig(render_path, dpi=150)
            plt.close()
            
            print(f"✓ Rendered views saved to: {render_path}")
            
            return True
            
        except Exception as e:
            print(f"✗ Rendering failed: {e}")
            traceback.print_exc()
            return False
    
    def test_full_pipeline(self, code: str, name: str) -> bool:
        """Test complete pipeline"""
        print(f"\n{'#'*80}")
        print(f"FULL PIPELINE TEST: {name}")
        print(f"{'#'*80}")
        
        results = {
            'code_to_mesh': False,
            'mesh_to_pointcloud': False,
            'rendering': False
        }
        
        # Test each component
        results['code_to_mesh'] = self.test_code_to_mesh(code, name)
        
        if results['code_to_mesh']:
            results['mesh_to_pointcloud'] = self.test_mesh_to_pointcloud(code, name)
            results['rendering'] = self.test_mesh_rendering(code, name)
        
        # Summary
        print(f"\n{'='*60}")
        print(f"SUMMARY for {name}:")
        print(f"{'='*60}")
        for test_name, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status}: {test_name}")
        
        all_passed = all(results.values())
        print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
        
        return all_passed


def load_test_scripts():
    """Load test CAD scripts from test_cad_scripts directory"""
    scripts_dir = "test_cad_scripts"
    
    if not os.path.exists(scripts_dir):
        print(f"Warning: {scripts_dir} not found. Using default test scripts.")
        return get_default_test_scripts()
    
    scripts = {}
    for file in Path(scripts_dir).glob("*.py"):
        name = file.stem
        with open(file, 'r') as f:
            scripts[name] = f.read()
    
    return scripts


def get_default_test_scripts():
    """Get default test scripts if directory doesn't exist"""
    return {
        "simple_cube": """import trimesh
import numpy as np

# Create a simple cube
vertices = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
])

faces = np.array([
    [0, 1, 2], [0, 2, 3],  # Bottom
    [4, 5, 6], [4, 6, 7],  # Top
    [0, 1, 5], [0, 5, 4],  # Front
    [2, 3, 7], [2, 7, 6],  # Back
    [0, 3, 7], [0, 7, 4],  # Left
    [1, 2, 6], [1, 6, 5]   # Right
])

mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
""",
        
        "sphere": """import trimesh

# Create a sphere using icosphere
mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.5)
""",
        
        "cylinder": """import trimesh

# Create a cylinder
mesh = trimesh.creation.cylinder(radius=1.0, height=3.0, sections=32)
""",
        
        "complex_shape": """import trimesh
import numpy as np

# Create a more complex shape - torus
mesh = trimesh.creation.annulus(r_min=0.5, r_max=1.5, height=0.3, sections=32)
"""
    }


def run_all_tests():
    """Run all mesh conversion tests"""
    print("=" * 80)
    print("MESH CONVERSION AND RENDERING TESTS")
    print("=" * 80)
    
    # Initialize tester
    tester = MeshConversionTester()
    
    # Load test scripts
    test_scripts = load_test_scripts()
    
    print(f"\nFound {len(test_scripts)} test scripts")
    print("Test scripts:", list(test_scripts.keys()))
    
    # Run tests for each script
    results = {}
    
    for name, code in test_scripts.items():
        results[name] = tester.test_full_pipeline(code, name)
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All mesh conversion tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)