import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import os
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
        
        print(f"Starting training from iteration {iteration}")
        
        while iteration < self.max_iterations:
            for batch in train_loader:
                if iteration >= self.max_iterations:
                    break
                
                # Generate samples
                prompts = batch['prompts']

                # Build prompts with system instruction
                full_prompts = [
                    SYSTEM_PROMPT
                    + "\nUser design description:\n"
                    + p
                    + "\n"
                    for p in prompts
                ]

                # Tokenize prompts
                prompt_encodings = self.model.tokenizer(
                    full_prompts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=128
                )
                
                input_ids = prompt_encodings['input_ids'].to(self.device)
                attention_mask = prompt_encodings['attention_mask'].to(self.device)
                
                # Generate CAD code
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.config.get('max_generation_length', 512),
                    temperature=self.config.get('temperature', 0.8),
                    top_k=self.config.get('top_k', 50),
                    top_p=self.config.get('top_p', 0.95)
                )
                
                # Decode and extract only the CAD script inside <script> tags
                raw_outputs = self.model.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )

                scripts: List[str] = [
                    extract_script_from_text(text) for text in raw_outputs
                ]
                
                # Compute rewards based only on the generated scripts
                rewards, detailed_rewards = self.reward_computer.compute_rewards(
                    scripts,
                    prompts=prompts
                )
                
                # Tokenize scripts to compute log probs and values only for script tokens
                script_encodings = self.model.tokenizer(
                    scripts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=self.config.get('max_generation_length', 512)
                )

                script_input_ids = script_encodings['input_ids'].to(self.device)
                script_attention_mask = script_encodings['attention_mask'].to(self.device)

                # Get log probs and values for scripts only
                log_probs, values = self.model.get_log_probs(
                    script_input_ids,
                    script_attention_mask
                )
                
                # Get reference log probs for KL penalty (scripts only)
                ref_log_probs = self.reference_model.get_log_probs(
                    script_input_ids,
                    script_attention_mask
                )
                
                # Compute KL divergence
                kl_div = (log_probs - ref_log_probs).sum(dim=1)
                
                # Adjust rewards with KL penalty
                adjusted_rewards = rewards - self.kl_coef * kl_div
                
                # Normalize rewards
                adjusted_rewards = self.reward_computer.normalize_rewards(adjusted_rewards)
                
                # Compute advantages
                advantages, returns = compute_advantages(
                    adjusted_rewards,
                    values,
                    gamma=self.gamma,
                    lam=self.lam
                )
                
                # PPO update
                ppo_losses = self.ppo_update(
                    script_input_ids,
                    script_attention_mask,
                    log_probs,
                    advantages,
                    returns,
                    values
                )
                
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
                    self.render_examples(generated_codes, iteration)
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
        total_losses = []
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.ppo_epochs):
            # Get current log probs and values
            current_log_probs, current_values = self.model.get_log_probs(input_ids, attention_mask)
            
            # Policy loss (PPO clipped objective)
            ratio = torch.exp(current_log_probs.sum(dim=1) - old_log_probs.sum(dim=1).detach())
            surr1 = ratio * advantages.detach()
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages.detach()
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(current_values, returns.detach())
            
            # Entropy loss (for exploration)
            outputs = self.model(input_ids, attention_mask)
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
            entropy_loss = -entropy
            
            # Total loss
            loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_losses.append(loss.item())
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy_loss.item())
        
        return {
            'total': torch.tensor(np.mean(total_losses)),
            'policy': torch.tensor(np.mean(policy_losses)),
            'value': torch.tensor(np.mean(value_losses)),
            'entropy': torch.tensor(np.mean(entropy_losses))
        }
    
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
                    max_length=512
                )
                
                raw_outputs = self.model.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )

                # Extract only the CAD script inside <script> tags
                scripts: List[str] = [
                    extract_script_from_text(text) for text in raw_outputs
                ]
                
                # Compute rewards based on scripts only
                rewards, _ = self.reward_computer.compute_rewards(scripts, prompts)
                total_rewards.append(rewards.mean().item())
                
                # Compute metrics on scripts
                for code in scripts:
                    metrics = compute_code_metrics(code)
                    code_metrics_list.append(metrics)
                
                # Compute perplexity on script tokens
                script_encodings = self.model.tokenizer(
                    scripts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                )

                script_input_ids = script_encodings['input_ids'].to(self.device)
                script_attention_mask = script_encodings['attention_mask'].to(self.device)

                outputs = self.model(
                    script_input_ids,
                    attention_mask=script_attention_mask,
                    labels=script_input_ids
                )
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
        
        # Policy loss (PPO clipped objective)
        ratio = torch.exp(current_log_probs.sum(dim=1) - old_log_probs.sum(dim=1).detach())
        surr1 = ratio * advantages.detach()
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages.detach()
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = nn.MSELoss()(current_values, returns.detach())
        
        # Entropy loss (for exploration)
        outputs = self.model(input_ids, attention_mask)
        logits = outputs['logits']
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
        entropy_loss = -entropy
        
        # Total loss
        loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        total_losses.append(loss.item())
        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())
        entropy_losses.append(entropy_loss.item())
    
    return {
        'total': torch.tensor(np.mean(total_losses)),
        'policy': torch.tensor(np.mean(policy_losses)),
        'value': torch.tensor(np.mean(value_losses)),
        'entropy': torch.tensor(np.mean(entropy_losses))
    }
    
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
                max_length=512
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


def main():
    """Main training script"""
    # Configuration
    config = {
        'model_name': 'Qwen/Qwen2.5-Coder-7B',
        'use_lora': True,
        'lora_r': 8,
        'lora_alpha': 16,
        'batch_size': 4,
        'learning_rate': 1e-5,
        'max_iterations': 10000,
        'ppo_epochs': 4,
        'clip_epsilon': 0.2,
        'value_loss_coef': 0.5,
        'entropy_coef': 0.01,
        'kl_coef': 0.1,
        'gamma': 0.99,
        'lam': 0.95,
        'max_generation_length': 512,
        'temperature': 0.8,
        'top_k': 50,
        'top_p': 0.95,
        'log_interval': 1,
        'eval_interval': 200,
        'save_interval': 500,
        'render_interval': 200,
        'checkpoint_dir': 'checkpoints',
        # UPDATED: Support for JSONL format
        'prompt_data_path': 'cadquery_prompts.jsonl',  # or 'data/prompts.json'
        'code_data_path': None,  # Optional
        'train_split': 0.9  # For JSONL files
    }
    
    # Initialize wandb
    wandb.init(
        project='cad-rl-training',
        config=config
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
    
    reward_computer = RewardComputer(
        reward_models=reward_models,
        device=device
    )
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
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