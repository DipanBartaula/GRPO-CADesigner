"""
Master test runner for all mesh conversion tests
Runs all mesh-related tests and generates a comprehensive report
"""

import sys
import os
import subprocess
import time
from datetime import datetime

def run_command(command, description):
    """Run a command and return success status"""
    print("\n" + "="*80)
    print(f"RUNNING: {description}")
    print("="*80)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=False,
            capture_output=False
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✓ {description} PASSED (took {elapsed:.2f}s)")
            return True
        else:
            print(f"\n✗ {description} FAILED (took {elapsed:.2f}s)")
            return False
            
    except Exception as e:
        print(f"\n✗ {description} CRASHED: {e}")
        return False


def setup_test_environment():
    """Setup test environment"""
    print("Setting up test environment...")
    
    # Create directories
    os.makedirs("test_outputs", exist_ok=True)
    os.makedirs("test_outputs/mesh_conversion", exist_ok=True)
    os.makedirs("test_outputs/visual_inspection", exist_ok=True)
    os.makedirs("test_cad_scripts", exist_ok=True)
    
    print("✓ Test directories created")
    
    # Generate simple test scripts
    try:
        exec(open("test_cad_scripts/create_simple_tests.py").read())
        print("✓ Test scripts generated")
    except FileNotFoundError:
        print("! Test script generator not found, will use defaults")
    except Exception as e:
        print(f"! Could not generate test scripts: {e}")


def main():
    """Run all mesh conversion tests"""
    print("="*80)
    print("MESH CONVERSION TEST SUITE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Setup
    setup_test_environment()
    
    # Define tests to run
    tests = [
        ("python test_mesh_conversion.py", "Mesh Conversion Tests"),
        ("python test_visual_inspection.py", "Visual Inspection Tests"),
    ]
    
    # Run all tests
    results = {}
    start_time = time.time()
    
    for command, description in tests:
        results[description] = run_command(command, description)
    
    total_time = time.time() - start_time
    
    # Generate summary report
    print("\n" + "="*80)
    print("FINAL TEST SUMMARY")
    print("="*80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} test suites passed")
    print(f"Total time: {total_time:.2f}s")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Output locations
    print("\n" + "="*80)
    print("OUTPUT LOCATIONS")
    print("="*80)
    print("📁 Mesh conversion outputs: test_outputs/mesh_conversion/")
    print("📁 Visual inspection reports: test_outputs/visual_inspection/")
    print("📁 Test CAD scripts: test_cad_scripts/")
    
    # Individual file listing
    try:
        print("\n" + "="*80)
        print("GENERATED FILES")
        print("="*80)
        
        mesh_files = os.listdir("test_outputs/mesh_conversion")
        if mesh_files:
            print("\nMesh Conversion Outputs:")
            for f in sorted(mesh_files):
                print(f"  • {f}")
        
        visual_files = os.listdir("test_outputs/visual_inspection")
        if visual_files:
            print("\nVisual Inspection Reports:")
            for f in sorted(visual_files):
                print(f"  • {f}")
        
        test_scripts = [f for f in os.listdir("test_cad_scripts") if f.endswith('.py')]
        if test_scripts:
            print("\nTest CAD Scripts:")
            for f in sorted(test_scripts):
                print(f"  • {f}")
    except Exception as e:
        print(f"Could not list files: {e}")
    
    print("\n" + "="*80)
    
    if passed == total:
        print("🎉 ALL MESH CONVERSION TESTS PASSED!")
        print("✓ Mesh to point cloud conversion working")
        print("✓ Point cloud normalization working")
        print("✓ Multi-view rendering working")
        print("✓ All visualization pipelines functional")
        return 0
    else:
        print(f"⚠️  {total - passed} test suite(s) failed")
        print("Please check the output above for details")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)