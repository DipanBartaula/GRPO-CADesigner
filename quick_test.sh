#!/bin/bash

# Quick test script - runs essential tests only
# Use this for fast validation during development

echo "=========================================="
echo "Quick Test Suite"
echo "=========================================="

# Create test data if needed
if [ ! -d "test_cad_scripts" ]; then
    echo "Creating test data..."
    python create_test_data.py
fi

# Run a subset of tests
echo ""
echo "Running quick validation tests..."
echo ""

# Test imports and basic functionality
python -c "
import sys
print('Testing imports...')
try:
    from model import PPOCADModel
    from reward_models import RewardModelEnsemble
    from dataloader import CADPromptDataset
    from utils import cad_code_to_mesh, mesh_to_point_cloud
    print('✓ All imports successful')
except Exception as e:
    print(f'✗ Import failed: {e}')
    sys.exit(1)

print('\nTesting mesh conversion...')
try:
    code = '''import trimesh
mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])'''
    mesh = cad_code_to_mesh(code)
    if mesh is not None:
        pc = mesh_to_point_cloud(mesh, 100)
        print(f'✓ Mesh conversion works: {len(mesh.vertices)} vertices, {pc.shape[0]} points')
    else:
        print('✗ Mesh conversion returned None')
        sys.exit(1)
except Exception as e:
    print(f'✗ Mesh conversion failed: {e}')
    sys.exit(1)

print('\n✓ Quick tests passed!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Quick tests PASSED"
    echo "=========================================="
    echo ""
    echo "For comprehensive tests, run:"
    echo "  python test.py"
    echo "  python run_all_mesh_tests.py"
else
    echo ""
    echo "=========================================="
    echo "✗ Quick tests FAILED"
    echo "=========================================="
    exit 1
fi