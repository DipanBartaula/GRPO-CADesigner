"""
Configuration file for CAD RL Training
"""

import os

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    # Base model to use (gpt2, gpt2-medium, microsoft/CodeGPT-small-py, etc.)
    'model_name': 'gpt2',
    
    # LoRA configuration
    'use_lora': True,
    'lora_r': 8,
    'lora_alpha': 16,
    'lora_dropout': 0.1,
    
    # Generation parameters
    'max_generation_length': 512,
    'temperature': 0.8,
    'top_k': 50,
    'top_p': 0.95,
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

TRAINING_CONFIG = {
    # Optimization
    'learning_rate': 1e-5,
    'batch_size': 4,
    'max_iterations': 10000,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    
    # PPO hyperparameters
    'ppo_epochs': 4,
    'clip_epsilon': 0.2,
    'value_loss_coef': 0.5,
    'entropy_coef': 0.01,
    'kl_coef': 0.1,
    'gamma': 0.99,  # Discount factor
    'lam': 0.95,    # GAE lambda
    
    # Logging and evaluation
    'log_interval': 1,           # Log every N iterations
    'eval_interval': 200,        # Eval every N seconds
    'save_interval': 500,        # Save checkpoint every N iterations
    'render_interval': 200,      # Render CAD objects every N iterations
    
    # Paths
    'checkpoint_dir': 'checkpoints',
    'output_dir': 'outputs',
    'log_dir': 'logs',
}

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

DATA_CONFIG = {
    'prompt_data_path': 'data/prompts.json',
    'code_data_path': 'data/code_examples.json',
    'num_workers': 2,
    'max_prompt_length': 128,
    'max_code_length': 512,
}

# ============================================================================
# REWARD MODEL CONFIGURATION
# ============================================================================

REWARD_MODEL_CONFIG = {
    # Reward model weights (must sum to 1.0)
    'reward_model_weights': {
        'pointbert': 0.2,
        'ulip2': 0.2,
        'multiview_clip': 0.2,
        'pointclip': 0.2,
        'geometric': 0.2,
    },
    
    # Enable/disable specific reward models
    'use_pointbert': True,
    'use_ulip2': True,
    'use_multiview_clip': True,
    'use_pointclip': True,
    'use_geometric': True,
    
    # Reward component weights
    'compilation_weight': 0.1,
    'execution_weight': 0.15,
    'cad_specific_weight': 0.15,
    'reward_model_weight': 0.6,
    
    # CAD-specific reward weights
    'cad_reward_weights': {
        'vertex_count': 0.1,
        'face_count': 0.1,
        'watertight': 0.3,
        'manifold': 0.2,
        'valid_topology': 0.2,
        'complexity': 0.1,
    },
}

# ============================================================================
# MESH PROCESSING CONFIGURATION
# ============================================================================

MESH_CONFIG = {
    'num_points': 2048,           # Number of points in point cloud
    'num_render_views': 4,        # Number of views to render
    'render_resolution': [224, 224],
    'execution_timeout': 5,       # Code execution timeout (seconds)
}

# ============================================================================
# WANDB CONFIGURATION
# ============================================================================

WANDB_CONFIG = {
    'project': 'cad-rl-training',
    'entity': None,  # Your WandB username or team
    'name': None,    # Run name (None for auto-generated)
    'notes': 'CAD generation with multiple reward models',
    'tags': ['rl', 'cad', 'generation', 'ppo'],
}

# ============================================================================
# HARDWARE CONFIGURATION
# ============================================================================

HARDWARE_CONFIG = {
    'device': 'cuda',  # 'cuda' or 'cpu'
    'mixed_precision': False,  # Use mixed precision training
    'num_gpus': 1,
}

# ============================================================================
# VALIDATION CONFIGURATION
# ============================================================================

def validate_config():
    """Validate configuration values"""
    errors = []
    
    # Check reward weights sum to 1.0
    reward_weights = REWARD_MODEL_CONFIG['reward_model_weights']
    if abs(sum(reward_weights.values()) - 1.0) > 1e-6:
        errors.append("Reward model weights must sum to 1.0")
    
    # Check CAD reward weights sum to 1.0
    cad_weights = REWARD_MODEL_CONFIG['cad_reward_weights']
    if abs(sum(cad_weights.values()) - 1.0) > 1e-6:
        errors.append("CAD reward weights must sum to 1.0")
    
    # Check positive values
    if TRAINING_CONFIG['learning_rate'] <= 0:
        errors.append("Learning rate must be positive")
    
    if TRAINING_CONFIG['batch_size'] <= 0:
        errors.append("Batch size must be positive")
    
    # Check paths exist
    os.makedirs(TRAINING_CONFIG['checkpoint_dir'], exist_ok=True)
    os.makedirs(TRAINING_CONFIG['output_dir'], exist_ok=True)
    os.makedirs(TRAINING_CONFIG['log_dir'], exist_ok=True)
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    print("✓ Configuration validated successfully")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_full_config():
    """Get merged configuration dictionary"""
    config = {}
    config.update(MODEL_CONFIG)
    config.update(TRAINING_CONFIG)
    config.update(DATA_CONFIG)
    config.update(REWARD_MODEL_CONFIG)
    config.update(MESH_CONFIG)
    config.update(WANDB_CONFIG)
    config.update(HARDWARE_CONFIG)
    return config

def print_config():
    """Print configuration in a readable format"""
    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    
    configs = [
        ("MODEL", MODEL_CONFIG),
        ("TRAINING", TRAINING_CONFIG),
        ("DATA", DATA_CONFIG),
        ("REWARD MODELS", REWARD_MODEL_CONFIG),
        ("MESH PROCESSING", MESH_CONFIG),
        ("WANDB", WANDB_CONFIG),
        ("HARDWARE", HARDWARE_CONFIG),
    ]
    
    for section_name, section_config in configs:
        print(f"\n{section_name}:")
        for key, value in section_config.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    validate_config()
    print_config()