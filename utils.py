import torch
import torch.nn as nn
import numpy as np
import os
import trimesh
import open3d as o3d
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import wandb

def xavier_init(model: nn.Module):
    """Initialize model with Xavier initialization"""
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
    return model

def load_checkpoint(model, optimizer, checkpoint_dir: str, device: torch.device):
    """Load the latest checkpoint"""
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoints:
        print("No checkpoint found. Initializing with Xavier...")
        xavier_init(model)
        return 0, model, optimizer
    
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    
    print(f"Resumed from iteration {iteration}")
    return iteration, model, optimizer

def save_checkpoint(model, optimizer, iteration: int, checkpoint_dir: str, metrics: Dict):
    """Save checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{iteration}.pt')
    
    torch.save({
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, checkpoint_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")

def cad_code_to_mesh(cad_code: str) -> Optional[trimesh.Trimesh]:
    """Execute CAD code and return mesh"""
    try:
        # Create a namespace for execution
        namespace = {}
        exec(cad_code, namespace)
        
        # Try to find mesh-like objects
        for key, value in namespace.items():
            if isinstance(value, trimesh.Trimesh):
                return value
            elif hasattr(value, 'export'):
                # Try to export to trimesh format
                try:
                    return trimesh.load_mesh(value)
                except:
                    continue
        return None
    except Exception as e:
        print(f"Error executing CAD code: {e}")
        return None

def mesh_to_point_cloud(mesh: trimesh.Trimesh, num_points: int = 2048) -> np.ndarray:
    """Sample point cloud from mesh"""
    if mesh is None:
        return np.zeros((num_points, 3))
    
    try:
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        return points
    except:
        return np.zeros((num_points, 3))

def render_mesh(mesh: trimesh.Trimesh, views: int = 4) -> List[np.ndarray]:
    """Render mesh from multiple views"""
    if mesh is None:
        return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(views)]
    
    rendered_images = []
    scene = mesh.scene()
    
    for i in range(views):
        angle = 2 * np.pi * i / views
        camera_pose = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
        camera_pose[:3, 3] = [2, 2, 2]
        scene.camera_transform = camera_pose
        
        try:
            png = scene.save_image(resolution=[224, 224])
            img = Image.open(BytesIO(png))
            rendered_images.append(np.array(img)[:, :, :3])
        except:
            rendered_images.append(np.zeros((224, 224, 3), dtype=np.uint8))
    
    return rendered_images

def log_rendered_mesh_to_wandb(mesh: trimesh.Trimesh, iteration: int, name: str = "cad_object"):
    """Log rendered CAD object to wandb"""
    if mesh is None:
        return
    
    views = render_mesh(mesh, views=4)
    
    # Create a figure with multiple views
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for idx, (ax, view) in enumerate(zip(axes.flat, views)):
        ax.imshow(view)
        ax.axis('off')
        ax.set_title(f'View {idx+1}')
    
    plt.tight_layout()
    wandb.log({f"{name}/rendered_views": wandb.Image(fig), "iteration": iteration})
    plt.close(fig)

def compute_perplexity(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
    """Compute perplexity from logits and labels"""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    perplexity = torch.exp(loss)
    return perplexity.item()

def tokenize_cad_code(code: str, tokenizer, max_length: int = 512) -> Dict:
    """Tokenize CAD code"""
    return tokenizer(
        code,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

def check_code_syntax(code: str) -> bool:
    """Check if code has valid Python syntax"""
    try:
        compile(code, '<string>', 'exec')
        return True
    except SyntaxError:
        return False

def compute_code_metrics(generated_code: str) -> Dict[str, float]:
    """Compute code quality metrics without ground truth"""
    metrics = {}
    
    # Syntax validity
    metrics['syntax_valid'] = float(check_code_syntax(generated_code))
    
    # Code length
    metrics['code_length'] = len(generated_code)
    metrics['num_lines'] = len(generated_code.split('\n'))
    
    # Keyword presence
    cad_keywords = ['import', 'trimesh', 'mesh', 'vertices', 'faces']
    metrics['keyword_coverage'] = sum(kw in generated_code.lower() for kw in cad_keywords) / len(cad_keywords)
    
    # Complexity (rough estimate)
    metrics['cyclomatic_complexity'] = generated_code.count('if') + generated_code.count('for') + generated_code.count('while')
    
    return metrics

def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    """Normalize point cloud to unit sphere"""
    centroid = np.mean(points, axis=0)
    points = points - centroid
    max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
    if max_dist > 0:
        points = points / max_dist
    return points

class EarlyStopping:
    """Early stopping handler"""
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, loss: float):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.counter = 0
        
        return self.early_stop