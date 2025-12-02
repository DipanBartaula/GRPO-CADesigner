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

def render_mesh_matplotlib(mesh: trimesh.Trimesh, views: int = 4) -> List[np.ndarray]:
    """
    Render mesh using matplotlib (simple but may have black background issues)
    Kept for backward compatibility
    """
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

def render_mesh_pyrender(mesh: trimesh.Trimesh, views: int = 4, resolution: Tuple[int, int] = (224, 224)) -> List[np.ndarray]:
    """
    Render mesh using pyrender with proper lighting
    FIXED: No more black images!
    """
    if mesh is None:
        return [np.zeros((resolution[0], resolution[1], 3), dtype=np.uint8) for _ in range(views)]
    
    try:
        import pyrender
        
        # Create pyrender mesh
        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
        
        rendered_images = []
        
        for i in range(views):
            # Create scene
            scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[255, 255, 255])
            scene.add(mesh_pyrender)
            
            # Add directional light
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
            
            # Calculate camera position for this view
            angle = 2 * np.pi * i / views
            distance = max(mesh.extents) * 2.5
            
            cam_x = distance * np.cos(angle)
            cam_y = distance * 0.5
            cam_z = distance * np.sin(angle)
            
            # Camera looking at the mesh center
            camera_pose = np.array([
                [np.cos(angle), 0, np.sin(angle), cam_x],
                [0, 1, 0, cam_y],
                [-np.sin(angle), 0, np.cos(angle), cam_z],
                [0, 0, 0, 1]
            ])
            
            scene.add(light, pose=camera_pose)
            
            # Camera
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
            scene.add(camera, pose=camera_pose)
            
            # Render
            renderer = pyrender.OffscreenRenderer(resolution[0], resolution[1])
            color, _ = renderer.render(scene)
            renderer.delete()
            
            rendered_images.append(color)
        
        return rendered_images
        
    except ImportError:
        print("Warning: pyrender not available, falling back to matplotlib rendering")
        return render_mesh_matplotlib(mesh, views)
    except Exception as e:
        print(f"Warning: pyrender rendering failed ({e}), falling back to matplotlib")
        return render_mesh_matplotlib(mesh, views)

def render_mesh_open3d(mesh: trimesh.Trimesh, views: int = 4, resolution: Tuple[int, int] = (224, 224)) -> List[np.ndarray]:
    """
    Render mesh using Open3D for better visualization
    Alternative rendering method with good lighting
    """
    if mesh is None:
        return [np.zeros((resolution[0], resolution[1], 3), dtype=np.uint8) for _ in range(views)]
    
    try:
        # Convert trimesh to open3d
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.faces)
        
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(triangles)
        
        # Compute normals
        mesh_o3d.compute_vertex_normals()
        
        # Paint mesh with a color
        mesh_o3d.paint_uniform_color([0.7, 0.7, 0.7])
        
        rendered_images = []
        
        # Setup visualizer (off-screen)
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=resolution[0], height=resolution[1], visible=False)
        vis.add_geometry(mesh_o3d)
        
        # Get view control
        view_control = vis.get_view_control()
        
        # Render from different angles
        for i in range(views):
            # Reset view
            vis.clear_geometries()
            vis.add_geometry(mesh_o3d)
            
            # Set camera
            angle = 2 * np.pi * i / views
            view_control.set_front([np.cos(angle), 0.3, np.sin(angle)])
            view_control.set_up([0, 1, 0])
            view_control.set_zoom(0.7)
            
            # Render
            vis.poll_events()
            vis.update_renderer()
            
            # Capture image
            image = np.asarray(vis.capture_screen_float_buffer(False))
            image = (image * 255).astype(np.uint8)
            
            rendered_images.append(image)
        
        vis.destroy_window()
        
        return rendered_images
        
    except Exception as e:
        print(f"Warning: Open3D rendering failed ({e}), falling back to pyrender")
        return render_mesh_pyrender(mesh, views, resolution)

def render_mesh(mesh: trimesh.Trimesh, views: int = 4, resolution: Tuple[int, int] = (224, 224), method: str = 'auto') -> List[np.ndarray]:
    """
    Render mesh from multiple views with proper lighting
    
    Args:
        mesh: Trimesh object
        views: Number of views
        resolution: Image resolution (width, height)
        method: 'auto', 'pyrender', 'open3d', or 'matplotlib'
    
    Returns:
        List of rendered images as numpy arrays
    """
    if method == 'auto':
        # Try pyrender first, then open3d, then matplotlib
        try:
            import pyrender
            return render_mesh_pyrender(mesh, views, resolution)
        except ImportError:
            try:
                import open3d as o3d
                return render_mesh_open3d(mesh, views, resolution)
            except:
                return render_mesh_matplotlib(mesh, views)
    elif method == 'pyrender':
        return render_mesh_pyrender(mesh, views, resolution)
    elif method == 'open3d':
        return render_mesh_open3d(mesh, views, resolution)
    elif method == 'matplotlib':
        return render_mesh_matplotlib(mesh, views)
    else:
        raise ValueError(f"Unknown rendering method: {method}")

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