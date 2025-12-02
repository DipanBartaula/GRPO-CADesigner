import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
import clip
import trimesh
from transformers import AutoModel, AutoTokenizer

class PointBERTRewardModel(nn.Module):
    """
    Point-BERT based reward model for evaluating point clouds
    Uses pretrained Point-BERT for feature extraction
    """
    def __init__(self, pretrained_path: Optional[str] = None):
        super().__init__()
        
        # Point cloud encoder (simplified Point-BERT architecture)
        self.point_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
        if pretrained_path:
            self.load_pretrained(pretrained_path)
    
    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        Args:
            point_cloud: (batch_size, num_points, 3)
        Returns:
            reward: (batch_size,)
        """
        # Encode points
        features = self.point_encoder(point_cloud)
        
        # Apply transformer
        features = self.transformer(features)
        
        # Global pooling
        global_features = torch.mean(features, dim=1)
        
        # Get reward
        reward = self.reward_head(global_features)
        return reward.squeeze(-1)
    
    def load_pretrained(self, path: str):
        try:
            state_dict = torch.load(path, map_location='cpu')
            self.load_state_dict(state_dict, strict=False)
            print(f"Loaded PointBERT weights from {path}")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")


class ULIP2RewardModel(nn.Module):
    """
    ULIP-2 based reward model for unified language-image-point understanding
    """
    def __init__(self, pretrained_path: Optional[str] = None):
        super().__init__()
        
        # Point cloud encoder
        self.point_encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 512, 1),
            nn.BatchNorm1d(512)
        )
        
        # Text encoder (using small transformer)
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True),
            num_layers=3
        )
        
        self.text_projection = nn.Linear(768, 512)  # For BERT embeddings
        
        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )
        
        if pretrained_path:
            self.load_pretrained(pretrained_path)
    
    def forward(
        self,
        point_cloud: torch.Tensor,
        text_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            point_cloud: (batch_size, num_points, 3)
            text_embeddings: (batch_size, seq_len, 768) optional
        Returns:
            reward: (batch_size,)
        """
        batch_size = point_cloud.shape[0]
        
        # Encode point cloud
        point_features = self.point_encoder(point_cloud.transpose(1, 2))
        point_features = torch.max(point_features, dim=2)[0]  # Global max pooling
        
        # Encode text if provided
        if text_embeddings is not None:
            text_features = self.text_projection(text_embeddings)
            text_features = self.text_encoder(text_features)
            text_features = torch.mean(text_features, dim=1)
        else:
            text_features = torch.zeros_like(point_features)
        
        # Concatenate features
        combined = torch.cat([point_features, text_features], dim=1)
        
        # Get reward
        reward = self.reward_head(combined)
        return reward.squeeze(-1)
    
    def load_pretrained(self, path: str):
        try:
            state_dict = torch.load(path, map_location='cpu')
            self.load_state_dict(state_dict, strict=False)
            print(f"Loaded ULIP-2 weights from {path}")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")


class MultiViewCLIPRewardModel(nn.Module):
    """
    Multi-view CLIP reward model
    Evaluates CAD objects from multiple rendered views
    """
    def __init__(self, clip_model_name: str = "ViT-B/32"):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model, self.preprocess = clip.load(clip_model_name, device=self.device)
        
        # Freeze CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
    
    def forward(self, images: torch.Tensor, text: Optional[List[str]] = None) -> torch.Tensor:
        """
        Args:
            images: (batch_size, num_views, C, H, W)
            text: List of text prompts
        Returns:
            reward: (batch_size,)
        """
        batch_size, num_views = images.shape[:2]
        
        # Reshape for CLIP
        images_flat = images.view(-1, *images.shape[2:])
        
        # Get image features
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images_flat)
        
        # Reshape back
        image_features = image_features.view(batch_size, num_views, -1)
        
        # Aggregate views
        aggregated_features = torch.mean(image_features, dim=1)
        
        # Get reward
        reward = self.reward_head(aggregated_features.float())
        return reward.squeeze(-1)


class PointCLIPRewardModel(nn.Module):
    """
    PointCLIP reward model combining point clouds with CLIP
    """
    def __init__(self, clip_model_name: str = "ViT-B/32"):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model, _ = clip.load(clip_model_name, device=self.device)
        
        # Freeze CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Point encoder
        self.point_encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 512, 1)
        )
        
        # Projection to CLIP space
        self.projection = nn.Linear(512, 512)
        
        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, point_cloud: torch.Tensor, text: Optional[List[str]] = None) -> torch.Tensor:
        """
        Args:
            point_cloud: (batch_size, num_points, 3)
            text: List of text prompts
        Returns:
            reward: (batch_size,)
        """
        # Encode point cloud
        point_features = self.point_encoder(point_cloud.transpose(1, 2))
        point_features = torch.max(point_features, dim=2)[0]
        
        # Project to CLIP space
        point_features = self.projection(point_features)
        
        # Get reward
        reward = self.reward_head(point_features)
        return reward.squeeze(-1)


class GeometricPlausibilityRewardModel(nn.Module):
    """
    Geometric plausibility reward model
    Checks for valid geometry, manifoldness, etc.
    """
    def __init__(self):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(10, 64),  # 10 geometric features
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        self.reward_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def extract_geometric_features(self, mesh: trimesh.Trimesh) -> torch.Tensor:
        """Extract geometric features from mesh"""
        features = []
        
        try:
            # Is watertight
            features.append(float(mesh.is_watertight))
            
            # Volume (normalized)
            volume = mesh.volume if mesh.is_watertight else 0.0
            features.append(min(volume / 10.0, 1.0))
            
            # Surface area (normalized)
            area = mesh.area
            features.append(min(area / 100.0, 1.0))
            
            # Number of faces (normalized)
            features.append(min(len(mesh.faces) / 10000.0, 1.0))
            
            # Number of vertices (normalized)
            features.append(min(len(mesh.vertices) / 10000.0, 1.0))
            
            # Euler characteristic
            euler = mesh.euler_number
            features.append(euler / 10.0)
            
            # Convexity
            try:
                convex_hull = mesh.convex_hull
                convexity = mesh.volume / convex_hull.volume if mesh.is_watertight else 0.0
                features.append(convexity)
            except:
                features.append(0.0)
            
            # Aspect ratio
            extents = mesh.extents
            aspect = max(extents) / (min(extents) + 1e-6)
            features.append(min(aspect / 10.0, 1.0))
            
            # Is valid
            features.append(float(mesh.is_valid))
            
            # Has duplicate faces
            features.append(1.0 - float(len(mesh.faces) != len(np.unique(mesh.faces, axis=0))))
            
        except Exception as e:
            # Return zero features if extraction fails
            features = [0.0] * 10
        
        return torch.tensor(features, dtype=torch.float32)
    
    def forward(self, geometric_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            geometric_features: (batch_size, 10)
        Returns:
            reward: (batch_size,)
        """
        features = self.feature_extractor(geometric_features)
        reward = self.reward_head(features)
        return reward.squeeze(-1)


class RewardModelEnsemble(nn.Module):
    """
    Ensemble of all reward models
    """
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        use_pointbert: bool = True,
        use_ulip2: bool = True,
        use_multiview_clip: bool = True,
        use_pointclip: bool = True,
        use_geometric: bool = True
    ):
        super().__init__()
        
        self.weights = weights or {
            'pointbert': 0.2,
            'ulip2': 0.2,
            'multiview_clip': 0.2,
            'pointclip': 0.2,
            'geometric': 0.2
        }
        
        # Initialize models
        self.models = nn.ModuleDict()
        
        if use_pointbert:
            self.models['pointbert'] = PointBERTRewardModel()
        
        if use_ulip2:
            self.models['ulip2'] = ULIP2RewardModel()
        
        if use_multiview_clip:
            self.models['multiview_clip'] = MultiViewCLIPRewardModel()
        
        if use_pointclip:
            self.models['pointclip'] = PointCLIPRewardModel()
        
        if use_geometric:
            self.models['geometric'] = GeometricPlausibilityRewardModel()
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            inputs: Dictionary containing different input modalities
        Returns:
            rewards: Dictionary of rewards from each model
        """
        rewards = {}
        
        for name, model in self.models.items():
            try:
                model_dtype = next(model.parameters()).dtype

                if name == 'pointbert' and 'point_cloud' in inputs:
                    rewards[name] = model(inputs['point_cloud'].to(dtype=model_dtype))
                elif name == 'ulip2' and 'point_cloud' in inputs:
                    text_emb = inputs.get('text_embeddings')
                    rewards[name] = model(
                        inputs['point_cloud'].to(dtype=model_dtype),
                        text_emb.to(dtype=model_dtype) if text_emb is not None else None,
                    )
                elif name == 'multiview_clip' and 'rendered_views' in inputs:
                    # CLIP handles its own dtype internally
                    rewards[name] = model(inputs['rendered_views'])
                elif name == 'pointclip' and 'point_cloud' in inputs:
                    rewards[name] = model(inputs['point_cloud'].to(dtype=model_dtype))
                elif name == 'geometric' and 'geometric_features' in inputs:
                    rewards[name] = model(inputs['geometric_features'].to(dtype=model_dtype))
            except Exception as e:
                print(f"Error in {name} reward model: {e}")
                rewards[name] = torch.zeros(1, device=next(model.parameters()).device)
        
        return rewards
    
    def compute_total_reward(self, rewards: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute weighted sum of rewards"""
        total = torch.zeros_like(list(rewards.values())[0])
        
        for name, reward in rewards.items():
            total += self.weights.get(name, 0.0) * reward
        
        return total