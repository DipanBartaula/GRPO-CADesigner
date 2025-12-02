import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from typing import List, Dict, Optional
import random

class CADPromptDataset(Dataset):
    """
    Dataset for CAD generation prompts
    """
    def __init__(
        self,
        data_path: str,
        max_length: int = 512,
        split: str = 'train'
    ):
        self.data_path = data_path
        self.max_length = max_length
        self.split = split
        
        self.prompts = self.load_prompts()
    
    def load_prompts(self) -> List[Dict]:
        """Load prompts from file"""
        prompts = []
        
        # Try to load from JSON file
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r') as f:
                data = json.load(f)
                prompts = data.get(self.split, [])
        else:
            # Generate default prompts if file doesn't exist
            print(f"Warning: Data file {self.data_path} not found. Using default prompts.")
            prompts = self.generate_default_prompts()
        
        return prompts
    
    def generate_default_prompts(self) -> List[Dict]:
        """Generate default CAD prompts for testing"""
        default_prompts = [
            {
                "prompt": "Generate a cube with side length 2.0",
                "description": "A simple cube"
            },
            {
                "prompt": "Create a sphere with radius 1.5",
                "description": "A sphere object"
            },
            {
                "prompt": "Generate a cylinder with radius 1.0 and height 3.0",
                "description": "A cylindrical object"
            },
            {
                "prompt": "Create a cone with base radius 1.5 and height 2.5",
                "description": "A cone shape"
            },
            {
                "prompt": "Generate a torus with major radius 2.0 and minor radius 0.5",
                "description": "A torus object"
            },
            {
                "prompt": "Create a rectangular box with dimensions 2x3x4",
                "description": "A rectangular box"
            },
            {
                "prompt": "Generate a pyramid with base side 2.0 and height 3.0",
                "description": "A pyramid shape"
            },
            {
                "prompt": "Create a hexagonal prism with side length 1.0 and height 2.0",
                "description": "A hexagonal prism"
            },
            {
                "prompt": "Generate a gear with 12 teeth",
                "description": "A gear mechanism"
            },
            {
                "prompt": "Create a table with 4 legs",
                "description": "A simple table"
            }
        ]
        
        return default_prompts
    
    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, idx: int) -> Dict:
        prompt_data = self.prompts[idx]
        
        return {
            'prompt': prompt_data.get('prompt', ''),
            'description': prompt_data.get('description', ''),
            'idx': idx
        }


class CADCodeDataset(Dataset):
    """
    Dataset for CAD code examples (for supervised pre-training)
    """
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        split: str = 'train'
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        
        self.examples = self.load_examples()
    
    def load_examples(self) -> List[Dict]:
        """Load CAD code examples"""
        examples = []
        
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r') as f:
                data = json.load(f)
                examples = data.get(self.split, [])
        else:
            print(f"Warning: Data file {self.data_path} not found. Using default examples.")
            examples = self.generate_default_examples()
        
        return examples
    
    def generate_default_examples(self) -> List[Dict]:
        """Generate default CAD code examples"""
        examples = [
            {
                "prompt": "Generate a cube",
                "code": """import trimesh
import numpy as np

# Create a cube
vertices = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
])

faces = np.array([
    [0, 1, 2], [0, 2, 3],
    [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4],
    [2, 3, 7], [2, 7, 6],
    [0, 3, 7], [0, 7, 4],
    [1, 2, 6], [1, 6, 5]
])

mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
"""
            },
            {
                "prompt": "Create a sphere",
                "code": """import trimesh

# Create a sphere
sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
mesh = sphere
"""
            },
            {
                "prompt": "Generate a cylinder",
                "code": """import trimesh

# Create a cylinder
cylinder = trimesh.creation.cylinder(radius=1.0, height=2.0, sections=32)
mesh = cylinder
"""
            }
        ]
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict:
        example = self.examples[idx]
        
        prompt = example.get('prompt', '')
        code = example.get('code', '')
        
        # Combine prompt and code
        full_text = f"# Prompt: {prompt}\n{code}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0).clone()
        }


def collate_prompt_batch(batch: List[Dict]) -> Dict:
    """Collate function for prompt dataset"""
    prompts = [item['prompt'] for item in batch]
    descriptions = [item['description'] for item in batch]
    indices = [item['idx'] for item in batch]
    
    return {
        'prompts': prompts,
        'descriptions': descriptions,
        'indices': indices
    }


def collate_code_batch(batch: List[Dict]) -> Dict:
    """Collate function for code dataset"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def create_dataloaders(
    prompt_data_path: str,
    code_data_path: str,
    tokenizer,
    batch_size: int = 4,
    num_workers: int = 2,
    max_length: int = 512
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for training
    
    Returns:
        train_prompt_loader: DataLoader for prompts (RL training)
        val_prompt_loader: DataLoader for validation prompts
        pretrain_loader: DataLoader for supervised pre-training
    """
    # Prompt datasets
    train_prompt_dataset = CADPromptDataset(
        data_path=prompt_data_path,
        max_length=max_length,
        split='train'
    )
    
    val_prompt_dataset = CADPromptDataset(
        data_path=prompt_data_path,
        max_length=max_length,
        split='val'
    )
    
    # Code dataset for pre-training
    pretrain_dataset = CADCodeDataset(
        data_path=code_data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        split='train'
    )
    
    # Create dataloaders
    train_prompt_loader = DataLoader(
        train_prompt_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_prompt_batch
    )
    
    val_prompt_loader = DataLoader(
        val_prompt_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_prompt_batch
    )
    
    pretrain_loader = DataLoader(
        pretrain_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_code_batch
    )
    
    return train_prompt_loader, val_prompt_loader, pretrain_loader


def save_example_data(
    prompt_path: str = "data/prompts.json",
    code_path: str = "data/code_examples.json"
):
    """Save example data files"""
    os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
    
    # Example prompts
    prompt_data = {
        "train": [
            {"prompt": "Generate a cube with side length 2.0", "description": "A simple cube"},
            {"prompt": "Create a sphere with radius 1.5", "description": "A sphere object"},
            {"prompt": "Generate a cylinder with radius 1.0 and height 3.0", "description": "A cylindrical object"},
        ] * 10,  # Repeat for more data
        "val": [
            {"prompt": "Create a cone with base radius 1.5 and height 2.5", "description": "A cone shape"},
            {"prompt": "Generate a torus with major radius 2.0 and minor radius 0.5", "description": "A torus object"},
        ] * 5
    }
    
    with open(prompt_path, 'w') as f:
        json.dump(prompt_data, f, indent=2)
    
    # Example code
    code_data = {
        "train": [
            {
                "prompt": "Generate a cube",
                "code": "import trimesh\nimport numpy as np\n\nvertices = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]])\nfaces = np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[2,3,7],[2,7,6],[0,3,7],[0,7,4],[1,2,6],[1,6,5]])\nmesh = trimesh.Trimesh(vertices=vertices, faces=faces)"
            }
        ] * 20
    }
    
    with open(code_path, 'w') as f:
        json.dump(code_data, f, indent=2)
    
    print(f"Saved example data to {prompt_path} and {code_path}")