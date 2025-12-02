# CAD Generation with Reinforcement Learning

A complete reinforcement learning training system for generating CAD objects using multiple reward models including PointBERT, ULIP-2, MultiView CLIP, PointCLIP, and geometric plausibility.

## Project Structure

```
.
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── model.py                  # Model architectures (PPO, Value Head, Reference Model)
├── reward_models.py          # All reward model implementations
├── rewards.py                # Reward computation logic
├── dataloader.py            # Data loading and processing
├── utils.py                 # Utility functions
├── train.py                 # Main training script
├── inference.py             # Inference and generation
├── test.py                  # Comprehensive test suite
├── checkpoints/             # Model checkpoints (auto-created)
├── data/                    # Training data (auto-created)
└── outputs/                 # Generated outputs (auto-created)
```

## Features

- **Multiple Reward Models**: 
  - PointBERT for point cloud evaluation
  - ULIP-2 for unified language-image-point understanding
  - MultiView CLIP for multi-view rendering evaluation
  - PointCLIP for point-text alignment
  - Geometric plausibility for topology validation
  
- **Code Quality Rewards**:
  - Compilation success/failure
  - Execution success/failure
  - CAD-specific metrics (watertight, manifold, vertex count, etc.)

- **Advanced Training**:
  - PPO (Proximal Policy Optimization) with GAE
  - LoRA support for efficient fine-tuning
  - Automatic checkpoint resume with Xavier initialization fallback
  - KL divergence penalty from reference model

- **Comprehensive Logging**:
  - WandB integration for all metrics
  - Per-iteration logging of losses and rewards
  - Evaluation metrics every 200 seconds
  - Rendered CAD objects logged every 200 iterations
  - Code quality metrics (perplexity, syntax validity, etc.)

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install CLIP
pip install git+https://github.com/openai/CLIP.git
```

## Quick Start

### 1. Create Test Data

Generate test CAD scripts and training data:

```bash
python create_test_data.py
```

This creates:
- 12 test CAD scripts in `test_cad_scripts/`
- Training prompts and code examples in `data/`

### 2. Run Tests

Verify everything is working:

```bash
# Test all core components
python test.py

# Test mesh conversion, point clouds, and rendering
python run_all_mesh_tests.py
```

**Core tests** (`test.py`):
- All imports
- Model initialization
- Reward models
- Reward computation
- Dataloaders
- Utility functions
- Forward passes
- Checkpoint operations

**Mesh tests** (`run_all_mesh_tests.py`):
- CAD code to mesh conversion
- Mesh to point cloud conversion
- Point cloud normalization
- Multi-view rendering (6 views)
- Visual inspection reports
- Tests on 12+ CAD scripts

### 2. Train the Model

```bash
# Initialize WandB (first time only)
wandb login

# Start training
python train.py
```

Training will:
- Auto-create example data if not found
- Load latest checkpoint or initialize with Xavier
- Log metrics to WandB every iteration
- Evaluate every 200 seconds
- Save checkpoints every 500 iterations
- Render CAD objects every 200 iterations

### 3. Generate CAD Objects

```bash
# Generate from trained model
python inference.py \
    --checkpoint checkpoints/checkpoint_1000.pt \
    --prompt "Generate a cube with rounded edges" \
    --output_dir outputs/cube
```

## Configuration

Edit the `config` dictionary in `train.py` to customize:

```python
config = {
    'model_name': 'gpt2',           # Base model
    'use_lora': True,               # Enable LoRA
    'lora_r': 8,                    # LoRA rank
    'batch_size': 4,                # Batch size
    'learning_rate': 1e-5,          # Learning rate
    'max_iterations': 10000,        # Total iterations
    'ppo_epochs': 4,                # PPO update epochs
    'clip_epsilon': 0.2,            # PPO clip epsilon
    'kl_coef': 0.1,                 # KL penalty coefficient
    'eval_interval': 200,           # Eval every N seconds
    'render_interval': 200,         # Render every N iterations
    # ... more options
}
```

## Reward Weights

Customize reward model weights in `train.py`:

```python
reward_models = RewardModelEnsemble(
    weights={
        'pointbert': 0.2,
        'ulip2': 0.2,
        'multiview_clip': 0.2,
        'pointclip': 0.2,
        'geometric': 0.2
    }
)
```

And reward component weights:

```python
reward_computer = RewardComputer(
    reward_models=reward_models,
    device=device,
    compilation_weight=0.1,
    execution_weight=0.15,
    cad_specific_weight=0.15,
    reward_model_weight=0.6
)
```

## Data Format

### Prompts (data/prompts.json)

```json
{
  "train": [
    {
      "prompt": "Generate a cube with side length 2.0",
      "description": "A simple cube"
    }
  ],
  "val": [
    {
      "prompt": "Create a sphere with radius 1.5",
      "description": "A sphere object"
    }
  ]
}
```

### Code Examples (data/code_examples.json)

```json
{
  "train": [
    {
      "prompt": "Generate a cube",
      "code": "import trimesh\n..."
    }
  ]
}
```

## Monitoring Training

### WandB Dashboard

Training automatically logs to WandB:
- Loss curves (total, policy, value, entropy)
- Reward metrics (total, per-component)
- Evaluation metrics (perplexity, code quality)
- Rendered CAD objects
- KL divergence

### Local Files

- Checkpoints: `checkpoints/checkpoint_*.pt`
- Generated outputs: `outputs/`
- Training logs: Console output

## Advanced Usage

### Custom Reward Models

Add your own reward model:

```python
class MyCustomReward(nn.Module):
    def __init__(self):
        super().__init__()
        # Your architecture
    
    def forward(self, inputs):
        # Compute reward
        return reward
```

Register in `RewardModelEnsemble`:

```python
self.models['my_custom'] = MyCustomReward()
```

### Distributed Training

The code supports multi-GPU training via PyTorch:

```python
model = nn.DataParallel(model)
```

### Custom Datasets

Create your own dataset:

```python
class MyCADDataset(Dataset):
    def __init__(self, data_path):
        self.data = load_your_data(data_path)
    
    def __getitem__(self, idx):
        return {
            'prompt': self.data[idx]['prompt'],
            'description': self.data[idx]['desc']
        }
```

## Troubleshooting

### CUDA Out of Memory

- Reduce `batch_size`
- Enable gradient checkpointing
- Use LoRA with smaller rank

### Checkpoint Not Loading

- Check checkpoint directory path
- Ensure checkpoint file exists
- Verify PyTorch version compatibility

### Reward Models Not Working

- Check if CLIP model downloads correctly
- Verify mesh generation produces valid meshes
- Enable debug logging in reward computation

### Low Training Speed

- Reduce `ppo_epochs`
- Disable rendering (`render_interval: -1`)
- Use smaller base model
- Enable mixed precision training

## Citation

If you use this code, please cite:

```bibtex
@software{cad_rl_training,
  title={CAD Generation with Reinforcement Learning},
  author={Your Name},
  year={2024}
}
```

## License

MIT License

## Contributing

Contributions welcome! Please:
1. Run tests: `python test.py`
2. Follow code style
3. Add tests for new features
4. Update documentation

## Acknowledgments

- PointBERT: Point Cloud Pre-training with BERT
- ULIP-2: Unified Language-Image-Point Understanding
- CLIP: Contrastive Language-Image Pre-training
- PPO: Proximal Policy Optimization
- Trimesh: Python library for mesh processing