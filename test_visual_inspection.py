"""
Visual inspection test for mesh conversion, point clouds, and rendering
Creates a comprehensive visual report with all test results
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys
from pathlib import Path
import traceback

from utils import (
    cad_code_to_mesh,
    mesh_to_point_cloud,
    render_mesh,
    normalize_point_cloud
)

class VisualInspector:
    """Create comprehensive visual inspection reports"""
    
    def __init__(self, output_dir: str = "test_outputs/visual_inspection"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_comprehensive_report(self, code: str, name: str):
        """Create a comprehensive visual report for a CAD script"""
        print(f"\nCreating comprehensive report for: {name}")
        print("=" * 60)
        
        try:
            # Convert code to mesh
            mesh = cad_code_to_mesh(code)
            
            if mesh is None:
                print("✗ Failed to create mesh")
                return False
            
            print(f"✓ Mesh created: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            
            # Generate point cloud
            point_cloud = mesh_to_point_cloud(mesh, num_points=2048)
            normalized_pc = normalize_point_cloud(point_cloud)
            
            print(f"✓ Point cloud generated: {point_cloud.shape}")
            
            # Render views
            rendered_views = render_mesh(mesh, views=6)
            
            print(f"✓ Rendered {len(rendered_views)} views")
            
            # Create comprehensive figure
            fig = plt.figure(figsize=(20, 12))
            gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
            
            # Title
            fig.suptitle(f'Comprehensive Report: {name}', fontsize=20, fontweight='bold')
            
            # Row 1: Mesh statistics and info
            ax_info = fig.add_subplot(gs[0, :2])
            ax_info.axis('off')
            
            info_text = f"""
MESH STATISTICS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Vertices:        {len(mesh.vertices):,}
• Faces:           {len(mesh.faces):,}
• Edges:           {len(mesh.edges):,}
• Watertight:      {mesh.is_watertight}
• Valid:           {mesh.is_valid}
• Volume:          {mesh.volume if mesh.is_watertight else 'N/A'}
• Surface Area:    {mesh.area:.4f}
• Bounding Box:    {mesh.bounds[0]} to {mesh.bounds[1]}
• Euler Number:    {mesh.euler_number}

POINT CLOUD STATISTICS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Points:          {len(point_cloud):,}
• Original Range:  X[{point_cloud[:,0].min():.3f}, {point_cloud[:,0].max():.3f}]
                   Y[{point_cloud[:,1].min():.3f}, {point_cloud[:,1].max():.3f}]
                   Z[{point_cloud[:,2].min():.3f}, {point_cloud[:,2].max():.3f}]
• Normalized Range: X[{normalized_pc[:,0].min():.3f}, {normalized_pc[:,0].max():.3f}]
                    Y[{normalized_pc[:,1].min():.3f}, {normalized_pc[:,1].max():.3f}]
                    Z[{normalized_pc[:,2].min():.3f}, {normalized_pc[:,2].max():.3f}]
            """
            
            ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            # Row 1: Original point cloud (3D)
            ax_pc_orig = fig.add_subplot(gs[0, 2], projection='3d')
            scatter = ax_pc_orig.scatter(
                point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
                c=point_cloud[:, 2], cmap='viridis', s=2, alpha=0.6
            )
            ax_pc_orig.set_title('Original Point Cloud', fontweight='bold')
            ax_pc_orig.set_xlabel('X')
            ax_pc_orig.set_ylabel('Y')
            ax_pc_orig.set_zlabel('Z')
            plt.colorbar(scatter, ax=ax_pc_orig, shrink=0.5)
            
            # Row 1: Normalized point cloud (3D)
            ax_pc_norm = fig.add_subplot(gs[0, 3], projection='3d')
            scatter2 = ax_pc_norm.scatter(
                normalized_pc[:, 0], normalized_pc[:, 1], normalized_pc[:, 2],
                c=normalized_pc[:, 2], cmap='plasma', s=2, alpha=0.6
            )
            ax_pc_norm.set_title('Normalized Point Cloud', fontweight='bold')
            ax_pc_norm.set_xlabel('X')
            ax_pc_norm.set_ylabel('Y')
            ax_pc_norm.set_zlabel('Z')
            plt.colorbar(scatter2, ax=ax_pc_norm, shrink=0.5)
            
            # Row 2 & 3: Rendered views
            view_positions = [
                (1, 0), (1, 1), (1, 2), (1, 3),
                (2, 0), (2, 1)
            ]
            
            view_names = ['Front', 'Right', 'Back', 'Left', 'Top', 'Bottom']
            
            for idx, ((row, col), view_name) in enumerate(zip(view_positions, view_names)):
                if idx < len(rendered_views):
                    ax = fig.add_subplot(gs[row, col])
                    ax.imshow(rendered_views[idx])
                    ax.axis('off')
                    ax.set_title(f'{view_name} View', fontweight='bold', fontsize=12)
            
            # Point cloud projections
            ax_proj_xy = fig.add_subplot(gs[2, 2])
            ax_proj_xy.scatter(normalized_pc[:, 0], normalized_pc[:, 1], 
                              c=normalized_pc[:, 2], cmap='viridis', s=1, alpha=0.5)
            ax_proj_xy.set_title('Point Cloud XY Projection', fontweight='bold')
            ax_proj_xy.set_xlabel('X')
            ax_proj_xy.set_ylabel('Y')
            ax_proj_xy.grid(True, alpha=0.3)
            ax_proj_xy.set_aspect('equal')
            
            ax_proj_xz = fig.add_subplot(gs[2, 3])
            ax_proj_xz.scatter(normalized_pc[:, 0], normalized_pc[:, 2],
                              c=normalized_pc[:, 1], cmap='plasma', s=1, alpha=0.5)
            ax_proj_xz.set_title('Point Cloud XZ Projection', fontweight='bold')
            ax_proj_xz.set_xlabel('X')
            ax_proj_xz.set_ylabel('Z')
            ax_proj_xz.grid(True, alpha=0.3)
            ax_proj_xz.set_aspect('equal')
            
            # Save report
            report_path = os.path.join(self.output_dir, f"{name}_comprehensive_report.png")
            plt.savefig(report_path, dpi=200, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Comprehensive report saved to: {report_path}")
            
            # Save mesh
            mesh_path = os.path.join(self.output_dir, f"{name}.obj")
            mesh.export(mesh_path)
            print(f"✓ Mesh saved to: {mesh_path}")
            
            # Save point cloud
            pc_path = os.path.join(self.output_dir, f"{name}_pointcloud.npy")
            np.save(pc_path, point_cloud)
            print(f"✓ Point cloud saved to: {pc_path}")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to create report: {e}")
            traceback.print_exc()
            return False


def load_test_scripts():
    """Load test scripts"""
    scripts_dir = "test_cad_scripts"
    
    if not os.path.exists(scripts_dir):
        print(f"Creating {scripts_dir} with default scripts...")
        # Create simple test scripts
        os.makedirs(scripts_dir, exist_ok=True)
        create_default_scripts(scripts_dir)
    
    scripts = {}
    for file in Path(scripts_dir).glob("*.py"):
        if file.name == 'create_simple_tests.py':
            continue
        name = file.stem
        with open(file, 'r') as f:
            scripts[name] = f.read()
    
    return scripts


def create_default_scripts(scripts_dir):
    """Create default test scripts"""
    default_scripts = {
        'simple_cube': '''import trimesh
mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
''',
        'sphere': '''import trimesh
mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.5)
''',
        'cylinder': '''import trimesh
mesh = trimesh.creation.cylinder(radius=1.0, height=3.0, sections=32)
''',
        'cone': '''import trimesh
mesh = trimesh.creation.cone(radius=1.5, height=3.0, sections=32)
'''
    }
    
    for name, code in default_scripts.items():
        filepath = os.path.join(scripts_dir, f"{name}.py")
        with open(filepath, 'w') as f:
            f.write(code)


def run_visual_inspection():
    """Run visual inspection for all test scripts"""
    print("=" * 80)
    print("VISUAL INSPECTION TEST")
    print("=" * 80)
    
    # Initialize inspector
    inspector = VisualInspector()
    
    # Load test scripts
    test_scripts = load_test_scripts()
    
    if not test_scripts:
        print("No test scripts found!")
        return 1
    
    print(f"\nFound {len(test_scripts)} test scripts")
    print("Scripts:", list(test_scripts.keys()))
    
    # Run visual inspection for each
    results = {}
    
    for name, code in test_scripts.items():
        print(f"\n{'='*80}")
        results[name] = inspector.create_comprehensive_report(code, name)
    
    # Summary
    print("\n" + "=" * 80)
    print("VISUAL INSPECTION SUMMARY")
    print("=" * 80)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} visual reports generated")
    print(f"Output directory: {inspector.output_dir}")
    
    if passed == total:
        print("\n🎉 All visual inspection tests passed!")
        print(f"📁 Check {inspector.output_dir} for detailed reports")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit_code = run_visual_inspection()
    sys.exit(exit_code)