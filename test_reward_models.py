"""
Test script for reward models
Tests all 5 reward models and their ensemble
"""

import sys
import torch
import numpy as np
from reward_models import (
    PointBERTRewardModel,
    ULIP2RewardModel,
    MultiViewCLIPRewardModel,
    PointCLIPRewardModel,
    GeometricPlausibilityRewardModel,
    RewardModelEnsemble
)
from utils import cad_code_to_mesh, mesh_to_point_cloud, render_mesh, normalize_point_cloud
import trimesh

def create_test_mesh():
    """Create a simple test mesh"""
    return trimesh.creation.box(extents=[2.0, 2.0, 2.0])

def test_pointbert_reward():
    """Test PointBERT reward model"""
    print("=" * 80)
    print("TEST 1: PointBERT Reward Model")
    print("=" * 80)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Initialize model
        model = PointBERTRewardModel().to(device)
        print("✓ Model initialized")
        
        # Create test point cloud
        mesh = create_test_mesh()
        point_cloud = mesh_to_point_cloud(mesh, num_points=1024)
        point_cloud = normalize_point_cloud(point_cloud)
        
        # Convert to tensor
        pc_tensor = torch.tensor(point_cloud, dtype=torch.float32, device=device).unsqueeze(0)
        
        print(f"  - Point cloud shape: {pc_tensor.shape}")
        
        # Forward pass
        reward = model(pc_tensor)
        
        print(f"✓ Forward pass successful")
        print(f"  - Reward shape: {reward.shape}")
        print(f"  - Reward value: {reward.item():.4f}")
        
        # Batch test
        batch_pc = pc_tensor.repeat(3, 1, 1)
        batch_reward = model(batch_pc)
        
        print(f"✓ Batch processing successful")
        print(f"  - Batch size: {batch_reward.shape[0]}")
        print(f"  - Batch rewards: {batch_reward.tolist()}")
        
        return True
        
    except Exception as e:
        print(f"✗ PointBERT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ulip2_reward():
    """Test ULIP-2 reward model"""
    print("\n" + "=" * 80)
    print("TEST 2: ULIP-2 Reward Model")
    print("=" * 80)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = ULIP2RewardModel().to(device)
        print("✓ Model initialized")
        
        # Create test inputs
        mesh = create_test_mesh()
        point_cloud = normalize_point_cloud(mesh_to_point_cloud(mesh, 1024))
        pc_tensor = torch.tensor(point_cloud, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Test without text embeddings
        reward = model(pc_tensor)
        print(f"✓ Forward pass (no text) successful")
        print(f"  - Reward: {reward.item():.4f}")
        
        # Test with text embeddings
        text_embeddings = torch.randn(1, 10, 768, device=device)
        reward_with_text = model(pc_tensor, text_embeddings)
        print(f"✓ Forward pass (with text) successful")
        print(f"  - Reward: {reward_with_text.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ ULIP-2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiview_clip_reward():
    """Test MultiView CLIP reward model"""
    print("\n" + "=" * 80)
    print("TEST 3: MultiView CLIP Reward Model")
    print("=" * 80)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = MultiViewCLIPRewardModel().to(device)
        print("✓ Model initialized")
        
        # Create test mesh and render
        mesh = create_test_mesh()
        views = render_mesh(mesh, views=4)
        
        print(f"  - Rendered {len(views)} views")
        
        # Convert to tensor
        views_array = np.stack(views)
        views_tensor = torch.tensor(
            views_array,
            dtype=torch.float32,
            device=device
        ).permute(0, 3, 1, 2).unsqueeze(0) / 255.0
        
        print(f"  - Views tensor shape: {views_tensor.shape}")
        
        # Forward pass
        reward = model(views_tensor)
        
        print(f"✓ Forward pass successful")
        print(f"  - Reward: {reward.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ MultiView CLIP test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pointclip_reward():
    """Test PointCLIP reward model"""
    print("\n" + "=" * 80)
    print("TEST 4: PointCLIP Reward Model")
    print("=" * 80)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = PointCLIPRewardModel().to(device)
        print("✓ Model initialized")
        
        # Create test point cloud
        mesh = create_test_mesh()
        point_cloud = normalize_point_cloud(mesh_to_point_cloud(mesh, 1024))
        pc_tensor = torch.tensor(point_cloud, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Forward pass
        reward = model(pc_tensor)
        
        print(f"✓ Forward pass successful")
        print(f"  - Reward: {reward.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ PointCLIP test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_geometric_reward():
    """Test Geometric Plausibility reward model"""
    print("\n" + "=" * 80)
    print("TEST 5: Geometric Plausibility Reward Model")
    print("=" * 80)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = GeometricPlausibilityRewardModel().to(device)
        print("✓ Model initialized")
        
        # Create test mesh
        mesh = create_test_mesh()
        
        # Extract geometric features
        features = model.extract_geometric_features(mesh)
        print(f"✓ Features extracted: {features.shape}")
        print(f"  - Features: {features.tolist()}")
        
        # Forward pass
        features_batch = features.unsqueeze(0).to(device)
        reward = model(features_batch)
        
        print(f"✓ Forward pass successful")
        print(f"  - Reward: {reward.item():.4f}")
        
        # Test with different meshes
        meshes = [
            trimesh.creation.box(extents=[1, 1, 1]),
            trimesh.creation.icosphere(radius=1.0),
            trimesh.creation.cylinder(radius=0.5, height=2.0),
        ]
        
        print("\nTesting multiple meshes:")
        for i, test_mesh in enumerate(meshes):
            feats = model.extract_geometric_features(test_mesh).unsqueeze(0).to(device)
            r = model(feats)
            print(f"  Mesh {i+1}: reward = {r.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Geometric reward test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reward_ensemble():
    """Test reward model ensemble"""
    print("\n" + "=" * 80)
    print("TEST 6: Reward Model Ensemble")
    print("=" * 80)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize ensemble
        ensemble = RewardModelEnsemble(
            weights={
                'pointbert': 0.2,
                'ulip2': 0.2,
                'multiview_clip': 0.2,
                'pointclip': 0.2,
                'geometric': 0.2
            }
        ).to(device)
        
        print("✓ Ensemble initialized")
        print(f"  - Models: {list(ensemble.models.keys())}")
        
        # Prepare inputs
        mesh = create_test_mesh()
        point_cloud = normalize_point_cloud(mesh_to_point_cloud(mesh, 2048))
        pc_tensor = torch.tensor(point_cloud, dtype=torch.float32, device=device).unsqueeze(0)
        
        views = render_mesh(mesh, views=4)
        views_tensor = torch.tensor(
            np.stack(views),
            dtype=torch.float32,
            device=device
        ).permute(0, 3, 1, 2).unsqueeze(0) / 255.0
        
        geometric_features = ensemble.models['geometric'].extract_geometric_features(mesh)
        geometric_features = geometric_features.unsqueeze(0).to(device)
        
        inputs = {
            'point_cloud': pc_tensor,
            'rendered_views': views_tensor,
            'geometric_features': geometric_features
        }
        
        # Compute rewards
        rewards = ensemble(inputs)
        
        print(f"\n✓ Ensemble forward pass successful")
        print(f"  Individual rewards:")
        for name, reward in rewards.items():
            print(f"    {name}: {reward.item():.4f}")
        
        # Compute total reward
        total_reward = ensemble.compute_total_reward(rewards)
        print(f"\n  Total weighted reward: {total_reward.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Ensemble test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_rewards():
    """Test batch reward computation"""
    print("\n" + "=" * 80)
    print("TEST 7: Batch Reward Computation")
    print("=" * 80)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        ensemble = RewardModelEnsemble().to(device)
        
        # Create multiple meshes
        meshes = [
            trimesh.creation.box(extents=[2, 2, 2]),
            trimesh.creation.icosphere(radius=1.5),
            trimesh.creation.cylinder(radius=1.0, height=3.0),
        ]
        
        print(f"Computing rewards for {len(meshes)} meshes...")
        
        all_rewards = []
        
        for i, mesh in enumerate(meshes):
            # Prepare inputs
            point_cloud = normalize_point_cloud(mesh_to_point_cloud(mesh, 2048))
            pc_tensor = torch.tensor(point_cloud, dtype=torch.float32, device=device).unsqueeze(0)
            
            views = render_mesh(mesh, views=4)
            views_tensor = torch.tensor(
                np.stack(views),
                dtype=torch.float32,
                device=device
            ).permute(0, 3, 1, 2).unsqueeze(0) / 255.0
            
            geometric_features = ensemble.models['geometric'].extract_geometric_features(mesh)
            geometric_features = geometric_features.unsqueeze(0).to(device)
            
            inputs = {
                'point_cloud': pc_tensor,
                'rendered_views': views_tensor,
                'geometric_features': geometric_features
            }
            
            # Compute rewards
            rewards = ensemble(inputs)
            total = ensemble.compute_total_reward(rewards)
            
            all_rewards.append(total.item())
            
            print(f"\n  Mesh {i+1} ({mesh.__class__.__name__}):")
            for name, reward in rewards.items():
                print(f"    {name}: {reward.item():.4f}")
            print(f"    TOTAL: {total.item():.4f}")
        
        print(f"\n✓ Batch computation successful")
        print(f"  - All rewards: {[f'{r:.4f}' for r in all_rewards]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Batch reward test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all reward model tests"""
    print("=" * 80)
    print("REWARD MODEL TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("PointBERT", test_pointbert_reward),
        ("ULIP-2", test_ulip2_reward),
        ("MultiView CLIP", test_multiview_clip_reward),
        ("PointCLIP", test_pointclip_reward),
        ("Geometric Plausibility", test_geometric_reward),
        ("Reward Ensemble", test_reward_ensemble),
        ("Batch Computation", test_batch_rewards),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All reward model tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)