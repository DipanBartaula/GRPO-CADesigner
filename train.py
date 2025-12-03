import os
# Disable tokenizer parallelism to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import wandb
import time
from typing import Dict, List
import numpy as np

from model import PPOCADModel, ReferenceModel, SYSTEM_PROMPT, extract_script_from_text
from reward_models import RewardModelEnsemble
from rewards import RewardComputer, compute_advantages
from dataloader import create_dataloaders, save_example_data
from utils import (
    load_checkpoint, save_checkpoint, compute_perplexity,
    compute_code_metrics, log_rendered_mesh_to_wandb, cad_code_to_mesh
)

class PPOTrainer:
    """PPO Trainer for CAD generation"""
    def __init__(
        self,
        model: PPOCADModel,
        reference_model: ReferenceModel,
        reward_computer: RewardComputer,
        optimizer: optim.Optimizer,
        config: Dict
    ):
        self.model = model
        self.reference_model = reference_model
        self.reward_computer = reward_computer
        self.optimizer = optimizer
        self.config = config
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.reference_model.to(self.device)
        
        # PPO hyperparameters
        self.ppo_epochs = config.get('ppo_epochs', 4)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.kl_coef = config.get('kl_coef', 0.1)
        self.gamma = config.get('gamma', 0.99)
        self.lam = config.get('lam', 0.95)
        
        # Training parameters
        self.max_iterations = config.get('max_iterations', 10000)
        self.log_interval = config.get('log_interval', 1)
        self.eval_interval = config.get('eval_interval', 200)
        self.save_interval = config.get('save_interval', 500)
        self.render_interval = config.get('render_interval', 200)
        
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Mixed-precision training - use FP32 for main models, FP16 for reward models
        # Disable AMP since main models are FP32, but reward models are FP16
        self.use_amp = False  # Disable AMP as main models are FP32
        self.scaler = GradScaler(enabled=self.use_amp)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        start_iteration: int = 0
    ):
        """Main training loop"""
        iteration = start_iteration
        last_eval_time = time.time()
        last_render_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"STARTING TRAINING FROM ITERATION {iteration}")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Using AMP: {self.use_amp}")
        print(f"Main models (Qwen): FP32")
        print(f"Reward models: FP16")
        print(f"Batch size: {self.config.get('batch_size', 'unknown')}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"Learning rate: {self.config.get('learning_rate', 'unknown')}")
        print(f"PPO epochs: {self.ppo_epochs}")
        print(f"Clip epsilon: {self.clip_epsilon}")
        print(f"{'='*60}\n")
        
        while iteration < self.max_iterations:
            iteration_start_time = time.time()
            print(f"\n--- ITERATION {iteration} ---")
            for batch_idx, batch in enumerate(train_loader):
                if iteration >= self.max_iterations:
                    break
                
                print(f"Batch {batch_idx}: Processing {len(batch['prompts'])} prompts")
                
                # Generate samples
                prompts = batch['prompts']
                print(f"Sample prompt: '{prompts[0][:100]}...'" if prompts else "No prompts found")

                # Build prompts with system instruction
                full_prompts = [
                    SYSTEM_PROMPT
                    + "\nUser design description:\n"
                    + p
                    + "\n"
                    for p in prompts
                ]

                # Tokenize prompts
                print(f"Tokenizing prompts...")
                prompt_encodings = self.model.tokenizer(
                    full_prompts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=128
                )
                
                input_ids = prompt_encodings['input_ids'].to(self.device)
                attention_mask = prompt_encodings['attention_mask'].to(self.device)
                print(f"Input IDs shape: {input_ids.shape}")
                print(f"Attention mask shape: {attention_mask.shape}")
                
                # Generate CAD code
                print(f"Generating CAD code with max_length={self.config.get('max_generation_length', 32768)}...")
                gen_start_time = time.time()
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.config.get('max_generation_length', 32768),
                    temperature=self.config.get('temperature', 0.8),
                    top_k=self.config.get('top_k', 50),
                    top_p=self.config.get('top_p', 0.95)
                )
                gen_time = time.time() - gen_start_time
                print(f"Generation completed in {gen_time:.2f}s")
                print(f"Generated IDs shape: {generated_ids.shape}")
                
                # Decode and extract only the CAD script inside <script> tags
                raw_outputs = self.model.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )

                # Debug: Print raw output info
                print(f"\n{'='*50}")
                print(f"RAW OUTPUT DEBUG INFO")
                print(f"{'='*50}")
                for idx, raw_out in enumerate(raw_outputs):
                    print(f"  Raw output [{idx}]: length={len(raw_out)} chars")
                    print(f"    First 300 chars: {raw_out[:300]}...")
                    print(f"    Last 200 chars: ...{raw_out[-200:]}")
                    has_script_open = '<script>' in raw_out
                    has_script_close = '</script>' in raw_out
                    print(f"    Contains <script> tag: {has_script_open}")
                    print(f"    Contains </script> tag: {has_script_close}")

                scripts: List[str] = [
                    extract_script_from_text(text) for text in raw_outputs
                ]
                
                # Debug: Print each script's length and status
                print(f"\n{'='*50}")
                print(f"EXTRACTED SCRIPTS DEBUG INFO")
                print(f"{'='*50}")
                print(f"Total scripts extracted: {len(scripts)}")
                for idx, script in enumerate(scripts):
                    script_len = len(script) if script else 0
                    is_empty = not script or not script.strip()
                    has_import = 'import' in script.lower() if script else False
                    has_cadquery = 'cadquery' in script.lower() if script else False
                    print(f"  Script [{idx}]:")
                    print(f"    Length: {script_len} chars")
                    print(f"    Is empty: {is_empty}")
                    print(f"    Has 'import': {has_import}")
                    print(f"    Has 'cadquery': {has_cadquery}")
                    if script and script.strip():
                        print(f"    Preview (first 200 chars): {script[:200]}...")
                    else:
                        print(f"    Preview: <EMPTY SCRIPT>")
                
                valid_scripts = [s for s in scripts if s.strip()]
                empty_scripts = [s for s in scripts if not s.strip()]
                print(f"\nSUMMARY:")
                print(f"  Valid (non-empty) scripts: {len(valid_scripts)}/{len(scripts)}")
                print(f"  Empty scripts: {len(empty_scripts)}/{len(scripts)}")
                if valid_scripts:
                    avg_len = sum(len(s) for s in valid_scripts) / len(valid_scripts)
                    print(f"  Average valid script length: {avg_len:.1f} chars")
                    print(f"  Min script length: {min(len(s) for s in valid_scripts)} chars")
                    print(f"  Max script length: {max(len(s) for s in valid_scripts)} chars")
                print(f"{'='*50}\n")
                
                # Compute rewards based only on the generated scripts
                print(f"Computing rewards for {len(scripts)} scripts...")
                reward_start_time = time.time()
                rewards, detailed_rewards = self.reward_computer.compute_rewards(
                    scripts,
                    prompts=prompts
                )
                reward_time = time.time() - reward_start_time
                print(f"Reward computation completed in {reward_time:.2f}s")
                print(f"Rewards shape: {rewards.shape}")
                print(f"Reward stats - Mean: {rewards.mean():.4f}, Std: {rewards.std():.4f}, Min: {rewards.min():.4f}, Max: {rewards.max():.4f}")
                
                # Print detailed rewards
                for key, values in detailed_rewards.items():
                    if values and len(values) > 0:
                        values_arr = torch.tensor(values) if isinstance(values[0], (int, float)) else values
                        print(f"  {key}: Mean={values_arr.mean():.4f}, Std={values_arr.std():.4f}")
                
                # Tokenize scripts to compute log probs and values only for script tokens
                script_encodings = self.model.tokenizer(
                    scripts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=self.config.get('max_generation_length', 32768)
                )

                script_input_ids = script_encodings['input_ids'].to(self.device)
                script_attention_mask = script_encodings['attention_mask'].to(self.device)

                # Get log probs and values for scripts only
                print(f"Computing log probabilities and values...")
                log_probs, values = self.model.get_log_probs(
                    script_input_ids,
                    script_attention_mask
                )
                print(f"Log probs shape: {log_probs.shape}")
                print(f"Values shape: {values.shape}")
                print(f"Log probs stats - Mean: {log_probs.mean():.4f}, Std: {log_probs.std():.4f}")
                print(f"Values stats - Mean: {values.mean():.4f}, Std: {values.std():.4f}")
                
                # Get reference log probs for KL penalty (scripts only)
                ref_log_probs = self.reference_model.get_log_probs(
                    script_input_ids,
                    script_attention_mask
                )
                
                # Compute KL divergence
                print(f"Computing KL divergence...")
                kl_div = (log_probs - ref_log_probs).sum(dim=1)
                print(f"KL divergence shape: {kl_div.shape}")
                print(f"KL divergence stats - Mean: {kl_div.mean():.4f}, Std: {kl_div.std():.4f}")
                
                # Adjust rewards with KL penalty
                adjusted_rewards = rewards - self.kl_coef * kl_div
                print(f"Adjusted rewards stats - Mean: {adjusted_rewards.mean():.4f}, Std: {adjusted_rewards.std():.4f}")
                
                # Debug: Check if rewards have variance
                print(f"\n{'='*50}")
                print(f"PRE-NORMALIZATION DEBUG")
                print(f"{'='*50}")
                print(f"Adjusted rewards (raw): {adjusted_rewards}")
                rewards_std = adjusted_rewards.std().item()
                if rewards_std < 1e-6:
                    print(f"⚠️ WARNING: Rewards have near-zero variance ({rewards_std:.8f})!")
                    print(f"   This means ALL scripts got nearly identical rewards.")
                    print(f"   PPO will have no learning signal (advantages will be ~0).")
                
                # Normalize rewards
                adjusted_rewards = self.reward_computer.normalize_rewards(adjusted_rewards)
                print(f"Normalized rewards: {adjusted_rewards}")
                
                # Compute advantages
                print(f"\n{'='*50}")
                print(f"ADVANTAGE COMPUTATION DEBUG")
                print(f"{'='*50}")
                print(f"Computing advantages with gamma={self.gamma}, lam={self.lam}...")
                advantages, returns = compute_advantages(
                    adjusted_rewards,
                    values,
                    gamma=self.gamma,
                    lam=self.lam
                )
                print(f"Advantages shape: {advantages.shape}, Returns shape: {returns.shape}")
                print(f"Advantages (raw): {advantages}")
                print(f"Returns (raw): {returns}")
                print(f"Advantages stats - Mean: {advantages.mean():.4f}, Std: {advantages.std():.4f}")
                print(f"Returns stats - Mean: {returns.mean():.4f}, Std: {returns.std():.4f}")
                
                adv_std = advantages.std().item()
                if adv_std < 1e-6:
                    print(f"⚠️ WARNING: Advantages have near-zero variance ({adv_std:.8f})!")
                    print(f"   PPO policy gradient will be ~0, model won't learn.")
                
                # Check for NaN/Inf in advantages
                if torch.isnan(advantages).any() or torch.isinf(advantages).any():
                    print(f"🚨 CRITICAL: NaN or Inf detected in advantages!")
                    print(f"   NaN count: {torch.isnan(advantages).sum().item()}")
                    print(f"   Inf count: {torch.isinf(advantages).sum().item()}")
                print(f"{'='*50}\\n")
                
                # PPO update
                print(f"Starting PPO update with {self.ppo_epochs} epochs...")
                ppo_start_time = time.time()
                ppo_losses = self.ppo_update(
                    script_input_ids,
                    script_attention_mask,
                    log_probs,
                    advantages,
                    returns,
                    values
                )
                ppo_time = time.time() - ppo_start_time
                print(f"PPO update completed in {ppo_time:.2f}s")
                print(f"PPO Losses - Total: {ppo_losses['total']:.4f}, Policy: {ppo_losses['policy']:.4f}, Value: {ppo_losses['value']:.4f}, Entropy: {ppo_losses['entropy']:.4f}")
                
                # Log to wandb
                if iteration % self.log_interval == 0:
                    self.log_metrics(
                        iteration,
                        ppo_losses,
                        rewards,
                        detailed_rewards,
                        kl_div
                    )
                
                # Evaluation
                current_time = time.time()
                if current_time - last_eval_time >= self.eval_interval:
                    self.evaluate(val_loader, iteration)
                    last_eval_time = current_time
                
                # Render CAD objects
                if current_time - last_render_time >= self.render_interval:
                    self.render_examples(scripts, iteration)
                    last_render_time = current_time
                
                # Save checkpoint
                if iteration % self.save_interval == 0 and iteration > 0:
                    metrics = {
                        'rewards': rewards.mean().item(),
                        'ppo_loss': ppo_losses['total'].item()
                    }
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        iteration,
                        self.checkpoint_dir,
                        metrics
                    )
                
                print(f"Iteration {iteration} completed successfully")
                print(f"Total iteration time: {time.time() - iteration_start_time:.2f}s")
                iteration += 1
        
        print("Training completed!")
    
    def ppo_update(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Perform PPO update"""
        print(f"\n  PPO UPDATE START")
        print(f"  Input shapes - IDs: {input_ids.shape}, Mask: {attention_mask.shape}")
        print(f"  Old log probs shape: {old_log_probs.shape}")
        print(f"  Advantages shape: {advantages.shape}, Returns shape: {returns.shape}")
        print(f"  Old values shape: {old_values.shape}")
        
        total_losses = []
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        # Normalize advantages
        advantages_mean = advantages.mean()
        advantages_std = advantages.std()
        print(f"  Advantages before normalization - Mean: {advantages_mean:.4f}, Std: {advantages_std:.4f}")
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        print(f"  Advantages after normalization - Mean: {advantages.mean():.4f}, Std: {advantages.std():.4f}")
        
        for epoch_idx in range(self.ppo_epochs):
            print(f"  PPO Epoch {epoch_idx + 1}/{self.ppo_epochs}")
            if self.use_amp:
                with autocast(dtype=torch.float16):
                    
                    # Get current log probs and values
                    current_log_probs, current_values = self.model.get_log_probs(input_ids, attention_mask)
                    print(f"    Current log probs shape: {current_log_probs.shape}, Current values shape: {current_values.shape}")

                    # Policy loss (PPO clipped objective)
                    ratio = torch.exp(current_log_probs.sum(dim=1) - old_log_probs.sum(dim=1).detach())
                    surr1 = ratio * advantages.detach()
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages.detach()
                    policy_loss = -torch.min(surr1, surr2).mean()
                    print(f"    Policy loss: {policy_loss.item():.4f}")

                    # Value loss
                    value_loss = nn.MSELoss()(current_values, returns.detach())
                    print(f"    Value loss: {value_loss.item():.4f}")

                    # Entropy loss (for exploration)
                    outputs = self.model(input_ids, attention_mask)
                    logits = outputs['logits']
                    probs = torch.softmax(logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
                    entropy_loss = -entropy
                    print(f"    Entropy loss: {entropy_loss.item():.4f}")

                    # Total loss
                    loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
                    print(f"    Total loss: {loss.item():.4f}")

                # Backward with GradScaler
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Unscale gradients before clipping
                self.scaler.unscale_(self.optimizer)
                
                # Check for NaN/Inf gradients before clipping and stepping
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                print(f"    Gradient norm: {grad_norm:.4f}")
                if torch.isfinite(grad_norm):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    print(f"    Optimizer step successful")
                else:
                    print(f"    Warning: NaN/Inf gradients detected, skipping optimizer step")
                    self.scaler.update()
            else:
                # FP32 fallback (CPU or no AMP)
                # Get current log probs and values
                current_log_probs, current_values = self.model.get_log_probs(input_ids, attention_mask)
                print(f"    Current log probs shape: {current_log_probs.shape}, Current values shape: {current_values.shape}")

                # Policy loss (PPO clipped objective)
                ratio = torch.exp(current_log_probs.sum(dim=1) - old_log_probs.sum(dim=1).detach())
                surr1 = ratio * advantages.detach()
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages.detach()
                policy_loss = -torch.min(surr1, surr2).mean()
                print(f"    Policy loss: {policy_loss.item():.4f}")

                # Value loss
                value_loss = nn.MSELoss()(current_values, returns.detach())
                print(f"    Value loss: {value_loss.item():.4f}")

                # Entropy loss (for exploration)
                outputs = self.model(input_ids, attention_mask)
                logits = outputs['logits']
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
                entropy_loss = -entropy
                print(f"    Entropy loss: {entropy_loss.item():.4f}")

                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
                print(f"    Total loss: {loss.item():.4f}")

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            total_losses.append(loss.item())
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy_loss.item())
            print(f"    Epoch {epoch_idx + 1} losses - Total: {loss.item():.4f}, Policy: {policy_loss.item():.4f}, Value: {value_loss.item():.4f}, Entropy: {entropy_loss.item():.4f}")
        
        avg_losses = {
            'total': torch.tensor(np.mean(total_losses)),
            'policy': torch.tensor(np.mean(policy_losses)),
            'value': torch.tensor(np.mean(value_losses)),
            'entropy': torch.tensor(np.mean(entropy_losses))
        }
        print(f"  PPO UPDATE COMPLETE - Avg losses: {avg_losses}")
        return avg_losses
    
    def evaluate(self, val_loader: DataLoader, iteration: int):
        """Evaluate on validation set"""
        self.model.eval()
        
        total_rewards = []
        perplexities = []
        code_metrics_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                prompts = batch['prompts']
                
                # Tokenize prompts
                prompt_encodings = self.model.tokenizer(
                    prompts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=128
                )
                
                input_ids = prompt_encodings['input_ids'].to(self.device)
                attention_mask = prompt_encodings['attention_mask'].to(self.device)
                
                # Generate
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.config.get('max_generation_length', 32768)
                )
                
                generated_codes = self.model.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )
                
                # Compute rewards
                rewards, _ = self.reward_computer.compute_rewards(generated_codes, prompts)
                total_rewards.append(rewards.mean().item())
                
                # Compute metrics
                for code in generated_codes:
                    metrics = compute_code_metrics(code)
                    code_metrics_list.append(metrics)
                
                # Compute perplexity
                outputs = self.model(generated_ids, attention_mask=(generated_ids != self.model.tokenizer.pad_token_id))
                if outputs['loss'] is not None:
                    perp = torch.exp(outputs['loss']).item()
                    perplexities.append(perp)
        
        # Log evaluation metrics
        avg_reward = np.mean(total_rewards)
        avg_perplexity = np.mean(perplexities) if perplexities else 0.0
        
        avg_code_metrics = {
            key: np.mean([m[key] for m in code_metrics_list])
            for key in code_metrics_list[0].keys()
        }
        
        wandb.log({
            'eval/reward': avg_reward,
            'eval/perplexity': avg_perplexity,
            **{f'eval/{k}': v for k, v in avg_code_metrics.items()},
            'iteration': iteration
        })
        
        self.model.train()
    
    def render_examples(self, generated_codes: List[str], iteration: int):
        """Render example CAD objects and log to wandb"""
        for i, code in enumerate(generated_codes[:3]):  # Render first 3 examples
            mesh = cad_code_to_mesh(code)
            if mesh is not None:
                log_rendered_mesh_to_wandb(mesh, iteration, name=f"example_{i}")
    
    def log_metrics(
        self,
        iteration: int,
        ppo_losses: Dict,
        rewards: torch.Tensor,
        detailed_rewards: Dict,
        kl_div: torch.Tensor
    ):
        """Log metrics to wandb"""
        metrics = {
            'iteration': iteration,
            'loss/total': ppo_losses['total'].item(),
            'loss/policy': ppo_losses['policy'].item(),
            'loss/value': ppo_losses['value'].item(),
            'loss/entropy': ppo_losses['entropy'].item(),
            'reward/total': rewards.mean().item(),
            'reward/std': rewards.std().item(),
            'reward/total_cum': rewards.sum().item(),
            'kl_divergence': kl_div.mean().item()
        }
        
        # Add detailed rewards
        for key, values in detailed_rewards.items():
            values_arr = np.array(values)
            metrics[f'reward/{key}_mean'] = float(values_arr.mean())
            metrics[f'reward/{key}_std'] = float(values_arr.std())
            metrics[f'reward/{key}_cum'] = float(values_arr.sum())
    
        wandb.log(metrics)
    """Main training script"""
    # Configuration
    config = {
        'model_name': 'Qwen/Qwen2.5-Coder-1.5B-Instruct',  # Instruct variant for better instruction following
        'use_lora': True,
        'lora_r': 128,
        'lora_alpha': 16,
        'batch_size': 128,
        'learning_rate': 1e-5,
        'max_iterations': 10000,
        'ppo_epochs': 4,
        'clip_epsilon': 0.2,
        'value_loss_coef': 0.5,
        'entropy_coef': 0.01,
        'kl_coef': 0.1,
        'gamma': 0.99,
        'lam': 0.95,
        'max_generation_length': 32768,
        'temperature': 0.8,
        'top_k': 50,
        'top_p': 0.95,
        'log_interval': 1,
        'eval_interval': 200,
        'save_interval': 200,
        'render_interval': 200,
        'checkpoint_dir': 'checkpoints',
        # UPDATED: Support for JSONL format
        'prompt_data_path': 'D:\GRPO-CADesigner\cadquery_prompts.jsonl',  # or 'data/prompts.json'
        'code_data_path': None,  # Optional
        'train_split': 0.9  # For JSONL files
    }
    

    WANDB_API_KEY = "08a4c57edfe8bc0393a2a7f093adf84e2a3b8986"
    WANDB_ENTITY = "078bct-anandi-tribhuvan-university-institute-of-engineering"

    # Initialize wandb
    wandb.login(key=WANDB_API_KEY)
    wandb.init(
        project="GRPOCADesigner",
        entity=WANDB_ENTITY,
        config=config,
    )
    
    # Print dataset info
    if os.path.exists(config['prompt_data_path']):
        from dataloader import print_dataset_stats
        print_dataset_stats(config['prompt_data_path'])
    else:
        print(f"Warning: Dataset not found at {config['prompt_data_path']}")
        print("Using default prompts...")
    
    # Initialize models
    print("Initializing models...")
    model = PPOCADModel(
        model_name=config['model_name'],
        use_lora=config['use_lora'],
        lora_r=config['lora_r'],
        lora_alpha=config['lora_alpha']
    )
    
    reference_model = ReferenceModel(model_name=config['model_name'])
    
    # Initialize reward models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_models = RewardModelEnsemble(
        weights={
            'pointbert': 0.2,
            'ulip2': 0.2,
            'multiview_clip': 0.2,
            'pointclip': 0.2,
            'geometric': 0.2
        }
    ).to(device)

    # Reward models are inference-only: use fp16 on GPU to reduce VRAM
    if device.type == "cuda":
        reward_models = reward_models.half()
    
    reward_computer = RewardComputer(
        reward_models=reward_models,
        device=device
    )
    
    # Optimizer (only trainable parameters, e.g., LoRA layers)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=config['learning_rate'])
    
    # Load checkpoint if available
    start_iteration, model, optimizer = load_checkpoint(
        model,
        optimizer,
        config['checkpoint_dir'],
        device
    )
    
    # Create dataloaders
    train_loader, val_loader, pretrain_loader = create_dataloaders(
        prompt_data_path=config['prompt_data_path'],
        code_data_path=config.get('code_data_path'),
        tokenizer=model.tokenizer,
        batch_size=config['batch_size'],
        train_split=config.get('train_split', 0.9)
    )
    
    # Initialize trainer
    trainer = PPOTrainer(
        model=model,
        reference_model=reference_model,
        reward_computer=reward_computer,
        optimizer=optimizer,
        config=config
    )
    
    # Start training
    print("Starting training...")
    trainer.train(train_loader, val_loader, start_iteration)
    
    wandb.finish()


if __name__ == '__main__':
    main()