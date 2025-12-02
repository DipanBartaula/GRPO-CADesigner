"""
Test script for model functionality
Tests model initialization, forward pass, generation
"""

import sys
import torch
import numpy as np
from model import PPOCADModel, ReferenceModel, ValueHead

def test_model_initialization():
    """Test model initialization"""
    print("=" * 80)
    print("TEST 1: Model Initialization")
    print("=" * 80)
    
    try:
        # Test PPO model without LoRA
        print("\nInitializing PPO model (no LoRA)...")
        model = PPOCADModel(model_name='gpt2', use_lora=False)
        
        print(f"✓ PPO model initialized")
        print(f"  - Vocab size: {model.generator.vocab_size}")
        print(f"  - Hidden size: {model.generator.hidden_size}")
        
        trainable, total = model.generator.get_trainable_parameters()
        print(f"  - Trainable params: {trainable:,}")
        print(f"  - Total params: {total:,}")
        
        # Test PPO model with LoRA
        print("\nInitializing PPO model (with LoRA)...")
        model_lora = PPOCADModel(
            model_name='gpt2',
            use_lora=True,
            lora_r=8,
            lora_alpha=16
        )
        
        print(f"✓ PPO model with LoRA initialized")
        trainable_lora, total_lora = model_lora.generator.get_trainable_parameters()
        print(f"  - Trainable params: {trainable_lora:,}")
        print(f"  - Total params: {total_lora:,}")
        print(f"  - Trainable ratio: {trainable_lora/total_lora*100:.2f}%")
        
        # Test reference model
        print("\nInitializing reference model...")
        ref_model = ReferenceModel(model_name='gpt2')
        
        print(f"✓ Reference model initialized")
        print(f"  - All parameters frozen: {all(not p.requires_grad for p in ref_model.parameters())}")
        
        # Test value head
        print("\nInitializing value head...")
        value_head = ValueHead(hidden_size=768)
        
        print(f"✓ Value head initialized")
        print(f"  - Parameters: {sum(p.numel() for p in value_head.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test forward pass through model"""
    print("\n" + "=" * 80)
    print("TEST 2: Forward Pass")
    print("=" * 80)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model = PPOCADModel(model_name='gpt2', use_lora=False).to(device)
        
        # Create dummy input
        batch_size = 2
        seq_len = 20
        input_ids = torch.randint(0, model.generator.vocab_size, (batch_size, seq_len)).to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
        
        print(f"\nInput shape: {input_ids.shape}")
        
        # Forward pass
        outputs = model(input_ids, attention_mask)
        
        print(f"✓ Forward pass successful")
        print(f"  - Logits shape: {outputs['logits'].shape}")
        print(f"  - Values shape: {outputs['values'].shape}")
        print(f"  - Hidden states shape: {outputs['hidden_states'].shape}")
        
        # Check outputs
        assert outputs['logits'].shape == (batch_size, seq_len, model.generator.vocab_size)
        assert outputs['values'].shape == (batch_size,)
        assert outputs['hidden_states'].shape == (batch_size, seq_len, model.generator.hidden_size)
        
        print(f"✓ Output shapes validated")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation():
    """Test code generation"""
    print("\n" + "=" * 80)
    print("TEST 3: Code Generation")
    print("=" * 80)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PPOCADModel(model_name='gpt2', use_lora=False).to(device)
        
        # Test prompts
        prompts = [
            "Generate a cube",
            "Create a sphere"
        ]
        
        print(f"Testing generation with {len(prompts)} prompts...")
        
        # Tokenize
        encodings = model.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        print(f"  - Input IDs shape: {input_ids.shape}")
        
        # Generate
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=200,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1
        )
        
        print(f"✓ Generation successful")
        print(f"  - Generated shape: {generated_ids.shape}")
        
        # Decode
        generated_texts = model.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        print(f"\n✓ Decoded {len(generated_texts)} generated texts")
        
        # Show examples
        print("\n" + "-" * 80)
        print("GENERATED SAMPLES:")
        print("-" * 80)
        
        for i, (prompt, generated) in enumerate(zip(prompts, generated_texts)):
            print(f"\nPrompt {i+1}: {prompt}")
            print(f"Generated:")
            print(generated[:300])
            if len(generated) > 300:
                print("...")
        
        return True
        
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_log_probs():
    """Test log probability computation"""
    print("\n" + "=" * 80)
    print("TEST 4: Log Probability Computation")
    print("=" * 80)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PPOCADModel(model_name='gpt2', use_lora=False).to(device)
        
        batch_size = 2
        seq_len = 15
        input_ids = torch.randint(0, model.generator.vocab_size, (batch_size, seq_len)).to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
        
        # Get log probs
        log_probs, values = model.get_log_probs(input_ids, attention_mask)
        
        print(f"✓ Log probs computed")
        print(f"  - Log probs shape: {log_probs.shape}")
        print(f"  - Values shape: {values.shape}")
        print(f"  - Log probs sample: {log_probs[0, :5]}")
        print(f"  - Values sample: {values}")
        
        # Verify shapes
        assert log_probs.shape == (batch_size, seq_len - 1)
        assert values.shape == (batch_size,)
        
        print(f"✓ Shapes validated")
        
        return True
        
    except Exception as e:
        print(f"✗ Log prob computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reference_model():
    """Test reference model for KL divergence"""
    print("\n" + "=" * 80)
    print("TEST 5: Reference Model")
    print("=" * 80)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        policy_model = PPOCADModel(model_name='gpt2', use_lora=False).to(device)
        ref_model = ReferenceModel(model_name='gpt2').to(device)
        
        batch_size = 2
        seq_len = 15
        input_ids = torch.randint(0, policy_model.generator.vocab_size, (batch_size, seq_len)).to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
        
        # Get log probs from both models
        policy_log_probs, _ = policy_model.get_log_probs(input_ids, attention_mask)
        ref_log_probs = ref_model.get_log_probs(input_ids, attention_mask)
        
        print(f"✓ Reference model log probs computed")
        print(f"  - Policy log probs shape: {policy_log_probs.shape}")
        print(f"  - Reference log probs shape: {ref_log_probs.shape}")
        
        # Compute KL divergence
        kl_div = (policy_log_probs - ref_log_probs).sum(dim=1)
        
        print(f"✓ KL divergence computed")
        print(f"  - KL div shape: {kl_div.shape}")
        print(f"  - KL div values: {kl_div}")
        
        return True
        
    except Exception as e:
        print(f"✗ Reference model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_generation():
    """Test batch generation with different prompts"""
    print("\n" + "=" * 80)
    print("TEST 6: Batch Generation with Multiple Prompts")
    print("=" * 80)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PPOCADModel(model_name='gpt2', use_lora=False).to(device)
        
        # More diverse prompts
        prompts = [
            "Create a solid block that is 100mm long, 50mm wide, and 20mm high",
            "Design a simple cylindrical shaft, 80mm long and 20mm in diameter",
            "Generate a flat washer with an outer diameter of 30mm",
        ]
        
        print(f"Generating code for {len(prompts)} prompts...")
        
        # Tokenize
        encodings = model.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        # Generate
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=256,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        
        generated_texts = model.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        print(f"\n✓ Batch generation successful")
        
        # Display results
        print("\n" + "-" * 80)
        print("BATCH GENERATION RESULTS:")
        print("-" * 80)
        
        for i, (prompt, generated) in enumerate(zip(prompts, generated_texts)):
            print(f"\n{'='*80}")
            print(f"PROMPT {i+1}:")
            print(f"{'='*80}")
            print(prompt)
            print(f"\n{'GENERATED CODE:':^80}")
            print("-" * 80)
            print(generated[:400])
            if len(generated) > 400:
                print("... [truncated]")
        
        return True
        
    except Exception as e:
        print(f"✗ Batch generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all model tests"""
    print("=" * 80)
    print("MODEL TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Model Initialization", test_model_initialization),
        ("Forward Pass", test_forward_pass),
        ("Code Generation", test_generation),
        ("Log Probability", test_log_probs),
        ("Reference Model", test_reference_model),
        ("Batch Generation", test_batch_generation),
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
        print("\n🎉 All model tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)