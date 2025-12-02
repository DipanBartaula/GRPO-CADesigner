"""
Test script for dataloader functionality
Tests JSONL loading, batching, and data pipeline
"""

import sys
import os
import torch
from dataloader import (
    CADPromptDataset,
    create_dataloaders,
    print_dataset_stats,
    collate_prompt_batch
)
from model import PPOCADModel

def test_jsonl_loading():
    """Test loading JSONL dataset"""
    print("=" * 80)
    print("TEST 1: JSONL Dataset Loading")
    print("=" * 80)
    
    jsonl_path = "cadquery_prompts.jsonl"
    
    if not os.path.exists(jsonl_path):
        print(f"✗ JSONL file not found: {jsonl_path}")
        print("  Please ensure cadquery_prompts.jsonl is in the current directory")
        return False
    
    try:
        # Load train split
        train_dataset = CADPromptDataset(
            data_path=jsonl_path,
            split='train',
            train_split=0.9
        )
        
        print(f"✓ Train dataset loaded: {len(train_dataset)} samples")
        
        # Load val split
        val_dataset = CADPromptDataset(
            data_path=jsonl_path,
            split='val',
            train_split=0.9
        )
        
        print(f"✓ Val dataset loaded: {len(val_dataset)} samples")
        
        # Test sample access
        sample = train_dataset[0]
        print(f"\n✓ Sample keys: {list(sample.keys())}")
        print(f"  - Prompt: {sample['prompt'][:80]}...")
        print(f"  - Difficulty: {sample['difficulty']}")
        print(f"  - Index: {sample['idx']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to load JSONL dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader_batching():
    """Test dataloader batching"""
    print("\n" + "=" * 80)
    print("TEST 2: Dataloader Batching")
    print("=" * 80)
    
    jsonl_path = "cadquery_prompts.jsonl"
    
    if not os.path.exists(jsonl_path):
        print(f"✗ JSONL file not found: {jsonl_path}")
        return False
    
    try:
        # Create model for tokenizer
        model = PPOCADModel(model_name='gpt2', use_lora=False)
        
        # Create dataloaders
        train_loader, val_loader, _ = create_dataloaders(
            prompt_data_path=jsonl_path,
            code_data_path=None,
            tokenizer=model.tokenizer,
            batch_size=4,
            num_workers=0,
            train_split=0.9
        )
        
        print(f"✓ Dataloaders created")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Val batches: {len(val_loader)}")
        
        # Test batch loading
        batch = next(iter(train_loader))
        
        print(f"\n✓ Batch loaded successfully")
        print(f"  - Batch keys: {list(batch.keys())}")
        print(f"  - Batch size: {len(batch['prompts'])}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataloader batching failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_content():
    """Test and print batch content"""
    print("\n" + "=" * 80)
    print("TEST 3: Batch Content Inspection")
    print("=" * 80)
    
    jsonl_path = "cadquery_prompts.jsonl"
    
    if not os.path.exists(jsonl_path):
        print(f"✗ JSONL file not found: {jsonl_path}")
        return False
    
    try:
        model = PPOCADModel(model_name='gpt2', use_lora=False)
        
        train_loader, _, _ = create_dataloaders(
            prompt_data_path=jsonl_path,
            code_data_path=None,
            tokenizer=model.tokenizer,
            batch_size=3,
            num_workers=0
        )
        
        # Get first 2 batches
        batches = []
        for i, batch in enumerate(train_loader):
            if i >= 2:
                break
            batches.append(batch)
        
        # Print batch 1
        print("\n" + "-" * 80)
        print("BATCH 1:")
        print("-" * 80)
        
        for i, (prompt, diff) in enumerate(zip(batches[0]['prompts'], batches[0]['difficulties'])):
            print(f"\nSample {i+1}:")
            print(f"  Difficulty: {diff}")
            print(f"  Prompt: {prompt[:150]}...")
            if len(prompt) > 150:
                print(f"          {prompt[150:300]}...")
        
        # Print batch 2
        print("\n" + "-" * 80)
        print("BATCH 2:")
        print("-" * 80)
        
        for i, (prompt, diff) in enumerate(zip(batches[1]['prompts'], batches[1]['difficulties'])):
            print(f"\nSample {i+1}:")
            print(f"  Difficulty: {diff}")
            print(f"  Prompt: {prompt[:150]}...")
            if len(prompt) > 150:
                print(f"          {prompt[150:300]}...")
        
        print(f"\n✓ Batch content inspection complete")
        
        return True
        
    except Exception as e:
        print(f"✗ Batch content inspection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tokenization():
    """Test tokenization of prompts"""
    print("\n" + "=" * 80)
    print("TEST 4: Prompt Tokenization")
    print("=" * 80)
    
    jsonl_path = "cadquery_prompts.jsonl"
    
    if not os.path.exists(jsonl_path):
        print(f"✗ JSONL file not found: {jsonl_path}")
        return False
    
    try:
        model = PPOCADModel(model_name='gpt2', use_lora=False)
        
        train_loader, _, _ = create_dataloaders(
            prompt_data_path=jsonl_path,
            code_data_path=None,
            tokenizer=model.tokenizer,
            batch_size=2,
            num_workers=0
        )
        
        batch = next(iter(train_loader))
        prompts = batch['prompts']
        
        # Tokenize
        encodings = model.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        
        print(f"✓ Tokenization successful")
        print(f"  - Input IDs shape: {encodings['input_ids'].shape}")
        print(f"  - Attention mask shape: {encodings['attention_mask'].shape}")
        
        # Show tokenization of first prompt
        print(f"\n  First prompt tokenization:")
        print(f"    Original: {prompts[0][:100]}...")
        print(f"    Tokens: {encodings['input_ids'][0][:20].tolist()}...")
        print(f"    Decoded: {model.tokenizer.decode(encodings['input_ids'][0][:50])}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Tokenization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_statistics():
    """Print dataset statistics"""
    print("\n" + "=" * 80)
    print("TEST 5: Dataset Statistics")
    print("=" * 80)
    
    jsonl_path = "cadquery_prompts.jsonl"
    
    if not os.path.exists(jsonl_path):
        print(f"✗ JSONL file not found: {jsonl_path}")
        return False
    
    try:
        print_dataset_stats(jsonl_path)
        return True
    except Exception as e:
        print(f"✗ Failed to print statistics: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all dataloader tests"""
    print("=" * 80)
    print("DATALOADER TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("JSONL Loading", test_jsonl_loading),
        ("Dataloader Batching", test_dataloader_batching),
        ("Batch Content", test_batch_content),
        ("Tokenization", test_tokenization),
        ("Dataset Statistics", test_dataset_statistics),
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
        print("\n🎉 All dataloader tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)