import torch
import numpy as np
import sys
import traceback
from typing import Dict, List

# Test imports
def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    try:
        import torch
        import transformers
        import trimesh
        import clip
        import wandb
        print("✓ All core imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_model_initialization():
    """Test model initialization"""
    print("\nTesting model initialization...")
    try:
        from model import PPOCADModel, ReferenceModel
        
        # Test PPO model
        model = PPOCADModel(model_name='gpt2', use_lora=False)
        print(f"✓ PPOCADModel initialized successfully")
        print(f"  - Vocab size: {model.generator.vocab_size}")
        print(f"  - Hidden size: {model.generator.hidden_size}")
        
        # Test reference model
        ref_model = ReferenceModel(model_name='gpt2')
        print(f"✓ ReferenceModel initialized successfully")
        
        return True
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        traceback.print_exc()
        return False


def test_reward_models():
    """Test reward model initialization"""
    print("\nTesting reward models...")
    try:
        from reward_models import (
            PointBERTRewardModel,
            ULIP2RewardModel,
            MultiViewCLIPRewardModel,
            PointCLIPRewardModel,
            GeometricPlausibilityRewardModel,
            RewardModelEnsemble
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test individual models
        pointbert = PointBERTRewardModel().to(device)
        print("✓ PointBERT model initialized")
        
        ulip2 = ULIP2RewardModel().to(device)
        print("✓ ULIP2 model initialized")
        
        multiview_clip = MultiViewCLIPRewardModel().to(device)
        print("✓ MultiView CLIP model initialized")
        
        pointclip = PointCLIPRewardModel().to(device)
        print("✓ PointCLIP model initialized")
        
        geometric = GeometricPlausibilityRewardModel().to(device)
        print("✓ Geometric plausibility model initialized")
        
        # Test ensemble
        ensemble = RewardModelEnsemble().to(device)
        print("✓ Reward model ensemble initialized")
        
        return True
    except Exception as e:
        print(f"✗ Reward model initialization failed: {e}")
        traceback.print_exc()
        return False


def test_reward_computation():
    """Test reward computation"""
    print("\nTesting reward computation...")
    try:
        from rewards import CodeCompilationReward, CodeExecutionReward, CADSpecificReward
        from reward_models import RewardModelEnsemble
        from rewards import RewardComputer
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize reward models
        reward_models = RewardModelEnsemble().to(device)
        reward_computer = RewardComputer(reward_models, device)
        
        # Test with simple CAD code
        test_codes = [
            """import trimesh
sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
mesh = sphere
""",
            """import trimesh
import numpy as np
vertices = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]])
faces = np.array([[0,1,2],[0,2,3]])
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
"""
        ]
        
        rewards, detailed_rewards = reward_computer.compute_rewards(test_codes)
        
        print(f"✓ Reward computation successful")
        print(f"  - Total rewards: {rewards}")
        print(f"  - Detailed rewards keys: {list(detailed_rewards.keys())}")
        
        return True
    except Exception as e:
        print(f"✗ Reward computation failed: {e}")
        traceback.print_exc()
        return False


def test_dataloader():
    """Test dataloader"""
    print("\nTesting dataloader...")
    try:
        from dataloader import CADPromptDataset, create_dataloaders, save_example_data
        from model import PPOCADModel
        
        # Create example data
        save_example_data()
        print("✓ Example data created")
        
        # Test dataset
        dataset = CADPromptDataset(
            data_path='data/prompts.json',
            split='train'
        )
        print(f"✓ Dataset loaded with {len(dataset)} samples")
        
        # Test sample
        sample = dataset[0]
        print(f"  - Sample keys: {list(sample.keys())}")
        print(f"  - Prompt: {sample['prompt'][:50]}...")
        
        # Test dataloaders
        model = PPOCADModel(model_name='gpt2', use_lora=False)
        train_loader, val_loader, pretrain_loader = create_dataloaders(
            prompt_data_path='data/prompts.json',
            code_data_path='data/code_examples.json',
            tokenizer=model.tokenizer,
            batch_size=2,
            num_workers=0
        )
        
        print(f"✓ Dataloaders created")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Val batches: {len(val_loader)}")
        print(f"  - Pretrain batches: {len(pretrain_loader)}")
        
        # Test batch
        batch = next(iter(train_loader))
        print(f"✓ Batch loaded successfully")
        print(f"  - Batch keys: {list(batch.keys())}")
        print(f"  - Number of prompts: {len(batch['prompts'])}")
        
        return True
    except Exception as e:
        print(f"✗ Dataloader test failed: {e}")
        traceback.print_exc()
        return False


def test_utils():
    """Test utility functions"""
    print("\nTesting utility functions...")
    try:
        from utils import (
            cad_code_to_mesh,
            mesh_to_point_cloud,
            render_mesh,
            check_code_syntax,
            compute_code_metrics,
            normalize_point_cloud
        )
        
        # Test code syntax checking
        valid_code = "import trimesh\nx = 1"
        invalid_code = "import trimesh\nx = ("
        
        assert check_code_syntax(valid_code) == True
        assert check_code_syntax(invalid_code) == False
        print("✓ Code syntax checking works")
        
        # Test code to mesh
        simple_code = """import trimesh
sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
mesh = sphere
"""
        mesh = cad_code_to_mesh(simple_code)
        if mesh is not None:
            print(f"✓ Code to mesh conversion works")
            print(f"  - Vertices: {len(mesh.vertices)}")
            print(f"  - Faces: {len(mesh.faces)}")
            
            # Test mesh to point cloud
            pc = mesh_to_point_cloud(mesh, num_points=1024)
            print(f"✓ Mesh to point cloud works")
            print(f"  - Point cloud shape: {pc.shape}")
            
            # Test point cloud normalization
            pc_norm = normalize_point_cloud(pc)
            print(f"✓ Point cloud normalization works")
            
            # Test rendering
            views = render_mesh(mesh, views=4)
            print(f"✓ Mesh rendering works")
            print(f"  - Number of views: {len(views)}")
            print(f"  - View shape: {views[0].shape}")
        else:
            print("! Mesh generation returned None (may be expected)")
        
        # Test code metrics
        metrics = compute_code_metrics(simple_code)
        print(f"✓ Code metrics computation works")
        print(f"  - Metrics: {list(metrics.keys())}")
        
        return True
    except Exception as e:
        print(f"✗ Utils test failed: {e}")
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test forward pass through models"""
    print("\nTesting forward pass...")
    try:
        from model import PPOCADModel
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PPOCADModel(model_name='gpt2', use_lora=False).to(device)
        
        # Create dummy input
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, model.generator.vocab_size, (batch_size, seq_len)).to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
        
        # Forward pass
        outputs = model(input_ids, attention_mask)
        
        print(f"✓ Forward pass successful")
        print(f"  - Logits shape: {outputs['logits'].shape}")
        print(f"  - Values shape: {outputs['values'].shape}")
        
        # Test generation
        generated = model.generate(
            input_ids=input_ids[:1],
            attention_mask=attention_mask[:1],
            max_length=50
        )
        
        print(f"✓ Generation successful")
        print(f"  - Generated shape: {generated.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass test failed: {e}")
        traceback.print_exc()
        return False


def test_checkpoint_operations():
    """Test checkpoint saving and loading"""
    print("\nTesting checkpoint operations...")
    try:
        from model import PPOCADModel
        from utils import save_checkpoint, load_checkpoint
        import torch.optim as optim
        import os
        import shutil
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create test directory
        test_checkpoint_dir = 'test_checkpoints'
        os.makedirs(test_checkpoint_dir, exist_ok=True)
        
        # Initialize model and optimizer
        model = PPOCADModel(model_name='gpt2', use_lora=False).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # Save checkpoint
        metrics = {'reward': 0.5, 'loss': 0.1}
        save_checkpoint(model, optimizer, 100, test_checkpoint_dir, metrics)
        print("✓ Checkpoint saved")
        
        # Load checkpoint
        new_model = PPOCADModel(model_name='gpt2', use_lora=False).to(device)
        new_optimizer = optim.Adam(new_model.parameters(), lr=1e-4)
        
        iteration, loaded_model, loaded_optimizer = load_checkpoint(
            new_model, new_optimizer, test_checkpoint_dir, device
        )
        
        print(f"✓ Checkpoint loaded")
        print(f"  - Loaded iteration: {iteration}")
        
        # Cleanup
        shutil.rmtree(test_checkpoint_dir)
        print("✓ Checkpoint operations successful")
        
        return True
    except Exception as e:
        print(f"✗ Checkpoint operations test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("=" * 80)
    print("RUNNING CAD RL PROJECT TESTS")
    print("=" * 80)
    
    tests = [
        ("Imports", test_imports),
        ("Model Initialization", test_model_initialization),
        ("Reward Models", test_reward_models),
        ("Reward Computation", test_reward_computation),
        ("Dataloader", test_dataloader),
        ("Utils", test_utils),
        ("Forward Pass", test_forward_pass),
        ("Checkpoint Operations", test_checkpoint_operations)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} test crashed: {e}")
            traceback.print_exc()
            results[test_name] = False
    
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
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)