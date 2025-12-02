"""
Test script specifically for rendering
Tests different rendering backends and fixes black image issue
"""

import sys
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from utils import (
    render_mesh,
    render_mesh_pyrender,
    render_mesh_open3d,
    render_mesh_matplotlib
)

def create_test_meshes():
    """Create various test meshes"""
    meshes = {
        'cube': trimesh.creation.box(extents=[2, 2, 2]),
        'sphere': trimesh.creation.icosphere(subdivisions=3, radius=1.5),
        'cylinder': trimesh.creation.cylinder(radius=1.0, height=3.0),
        'cone': trimesh.creation.cone(radius=1.5, height=2.5),
        'torus': trimesh.creation.annulus(r_min=0.5, r_max=1.5, height=0.3),
    }
    return meshes

def test_rendering_backend(backend_name, render_func, mesh):
    """Test a specific rendering backend"""
    print(f"\nTesting {backend_name}...")
    
    try:
        views = render_func(mesh, views=4)
        
        # Check if images are not all black
        all_black = all(np.mean(view) < 5 for view in views)
        
        if all_black:
            print(f"  ⚠️  {backend_name}: Images are black (lighting issue)")
            return False, views
        else:
            print(f"  ✓ {backend_name}: Rendering successful")
            print(f"    - Mean pixel values: {[np.mean(v) for v in views]}")
            return True, views
            
    except Exception as e:
        print(f"  ✗ {backend_name}: Failed - {e}")
        return False, None


def test_all_backends():
    """Test all rendering backends"""
    print("=" * 80)
    print("RENDERING BACKEND TESTS")
    print("=" * 80)
    
    meshes = create_test_meshes()
    test_mesh = meshes['cube']
    
    results = {}
    rendered_views = {}
    
    # Test pyrender
    print("\n1. Testing PyRender (recommended)...")
    success, views = test_rendering_backend(
        "PyRender",
        render_mesh_pyrender,
        test_mesh
    )
    results['pyrender'] = success
    if views:
        rendered_views['pyrender'] = views
    
    # Test open3d
    print("\n2. Testing Open3D...")
    success, views = test_rendering_backend(
        "Open3D",
        render_mesh_open3d,
        test_mesh
    )
    results['open3d'] = success
    if views:
        rendered_views['open3d'] = views
    
    # Test matplotlib
    print("\n3. Testing Matplotlib...")
    success, views = test_rendering_backend(
        "Matplotlib",
        render_mesh_matplotlib,
        test_mesh
    )
    results['matplotlib'] = success
    if views:
        rendered_views['matplotlib'] = views
    
    # Test auto method
    print("\n4. Testing Auto (best available)...")
    success, views = test_rendering_backend(
        "Auto",
        lambda m, v: render_mesh(m, views=v, method='auto'),
        test_mesh
    )
    results['auto'] = success
    if views:
        rendered_views['auto'] = views
    
    return results, rendered_views


def visualize_results(rendered_views, output_path='test_outputs/rendering_comparison.png'):
    """Visualize rendering results from all backends"""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    backends = list(rendered_views.keys())
    n_backends = len(backends)
    
    if n_backends == 0:
        print("No rendered views to visualize")
        return
    
    fig, axes = plt.subplots(n_backends, 4, figsize=(16, 4 * n_backends))
    
    if n_backends == 1:
        axes = axes.reshape(1, -1)
    
    for i, backend in enumerate(backends):
        views = rendered_views[backend]
        for j, view in enumerate(views):
            axes[i, j].imshow(view)
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_title(f'{backend.upper()}\nView {j+1}', fontweight='bold')
            else:
                axes[i, j].set_title(f'View {j+1}')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison saved to: {output_path}")
    plt.close()


def test_mesh_rendering():
    """Test rendering for multiple meshes"""
    print("\n" + "=" * 80)
    print("MULTI-MESH RENDERING TEST")
    print("=" * 80)
    
    meshes = create_test_meshes()
    
    for name, mesh in meshes.items():
        print(f"\nRendering {name}...")
        
        try:
            views = render_mesh(mesh, views=4, method='auto')
            
            mean_values = [np.mean(v) for v in views]
            all_black = all(mv < 5 for mv in mean_values)
            
            if all_black:
                print(f"  ⚠️  {name}: Images are black")
            else:
                print(f"  ✓ {name}: Successful")
                print(f"    - Mean values: {[f'{mv:.1f}' for mv in mean_values]}")
            
            # Save visualization
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            for i, (ax, view) in enumerate(zip(axes, views)):
                ax.imshow(view)
                ax.axis('off')
                ax.set_title(f'View {i+1}')
            
            plt.suptitle(f'{name.upper()} - Rendered Views', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            import os
            os.makedirs('test_outputs/rendering', exist_ok=True)
            plt.savefig(f'test_outputs/rendering/{name}_rendered.png', dpi=150)
            plt.close()
            
            print(f"    - Saved to: test_outputs/rendering/{name}_rendered.png")
            
        except Exception as e:
            print(f"  ✗ {name}: Failed - {e}")


def test_rendering_parameters():
    """Test different rendering parameters"""
    print("\n" + "=" * 80)
    print("RENDERING PARAMETERS TEST")
    print("=" * 80)
    
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.5)
    
    # Test different numbers of views
    for n_views in [2, 4, 6, 8]:
        print(f"\nTesting {n_views} views...")
        try:
            views = render_mesh(mesh, views=n_views, method='auto')
            print(f"  ✓ Rendered {len(views)} views successfully")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    # Test different resolutions
    for resolution in [(128, 128), (224, 224), (512, 512)]:
        print(f"\nTesting resolution {resolution}...")
        try:
            views = render_mesh(mesh, views=4, resolution=resolution, method='auto')
            print(f"  ✓ Rendered at {resolution}: {views[0].shape}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")


def run_all_tests():
    """Run all rendering tests"""
    print("=" * 80)
    print("COMPREHENSIVE RENDERING TEST SUITE")
    print("=" * 80)
    
    # Test 1: Backend comparison
    results, rendered_views = test_all_backends()
    
    # Visualize comparison
    if rendered_views:
        visualize_results(rendered_views)
    
    # Test 2: Multiple meshes
    test_mesh_rendering()
    
    # Test 3: Parameters
    test_rendering_parameters()
    
    # Summary
    print("\n" + "=" * 80)
    print("RENDERING TEST SUMMARY")
    print("=" * 80)
    
    print("\nBackend Results:")
    for backend, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {backend}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if results.get('pyrender', False):
        print("\n✓ PyRender is working correctly (RECOMMENDED)")
        print("  This backend provides the best lighting and rendering quality.")
        print("  No black images!")
    elif results.get('open3d', False):
        print("\n✓ Open3D is working (GOOD ALTERNATIVE)")
        print("  Use this if PyRender is not available.")
    else:
        print("\n⚠️  No optimal backend found")
        print("  Install PyRender: pip install pyrender pyglet<2.0")
    
    print("\n" + "=" * 80)
    
    return 0 if any(results.values()) else 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)