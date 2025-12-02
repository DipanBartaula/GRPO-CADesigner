import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from typing import List, Dict, Optional
import random

class CADPromptDataset(Dataset):
    """
    Dataset for CAD generation prompts
    UPDATED: Now supports JSONL format
    """
    def __init__(
        self,
        data_path: str,
        max_length: int = 512,
        split: str = 'train',
        train_split: float = 0.9
    ):
        self.data_path = data_path
        self.max_length = max_length
        self.split = split
        self.train_split = train_split
        
        self.prompts = self.load_prompts()
    
    def load_prompts(self) -> List[Dict]:
        """Load prompts from JSONL or JSON file"""
        prompts = []
        
        if not os.path.exists(self.data_path):
            print(f"Warning: Data file {self.data_path} not found. Using default prompts.")
            return self.generate_default_prompts()
        
        # Check file extension
        if self.data_path.endswith('.jsonl'):
            prompts = self.load_jsonl()
        elif self.data_path.endswith('.json'):
            prompts = self.load_json()
        else:
            raise ValueError(f"Unsupported file format: {self.data_path}. Use .json or .jsonl")
        
        return prompts
    
    def load_jsonl(self) -> List[Dict]:
        """Load prompts from JSONL file"""
        print(f"Loading JSONL dataset from: {self.data_path}")
        
        all_prompts = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    all_prompts.append(data)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
        
        print(f"Loaded {len(all_prompts)} prompts from JSONL")
        
        # Split into train/val
        if self.split == 'train':
            split_idx = int(len(all_prompts) * self.train_split)
            prompts = all_prompts[:split_idx]
            print(f"Using {len(prompts)} prompts for training")
        elif self.split == 'val':
            split_idx = int(len(all_prompts) * self.train_split)
            prompts = all_prompts[split_idx:]
            print(f"Using {len(prompts)} prompts for validation")
        else:
            prompts = all_prompts
            print(f"Using all {len(prompts)} prompts")
        
        return prompts
    
    def load_json(self) -> List[Dict]:
        """Load prompts from JSON file (original format)"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            prompts = data.get(self.split, [])
        
        print(f"Loaded {len(prompts)} prompts from JSON ({self.split} split)")
        return prompts
    
    def generate_default_prompts(self) -> List[Dict]:
        """Generate default CAD prompts for testing"""
        default_prompts = [
            {
                "prompt": "Generate a cube with side length 2.0",
                "difficulty": "basic"
            },
            {
                "prompt": "Create a sphere with radius 1.5",
                "difficulty": "basic"
            },
            {
                "prompt": "Generate a cylinder with radius 1.0 and height 3.0",
                "difficulty": "basic"
            },
            {
                "prompt": "Create a cone with base radius 1.5 and height 2.5",
                "difficulty": "intermediate"
            },
            {
                "prompt": "Generate a torus with major radius 2.0 and minor radius 0.5",
                "difficulty": "intermediate"
            },
        ]
        
        return default_prompts
    
    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, idx: int) -> Dict:
        prompt_data = self.prompts[idx]
        
        return {
            'prompt': prompt_data.get('prompt', ''),
            'difficulty': prompt_data.get('difficulty', 'unknown'),
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
            with open(self.data_path, 'r', encoding='utf-8') as f:
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
    difficulties = [item.get('difficulty', 'unknown') for item in batch]
    indices = [item['idx'] for item in batch]
    
    return {
        'prompts': prompts,
        'difficulties': difficulties,
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
    code_data_path: Optional[str],
    tokenizer,
    batch_size: int = 4,
    num_workers: int = 2,
    max_length: int = 512,
    train_split: float = 0.9
) -> tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    # Prompt datasets
    train_prompt_dataset = CADPromptDataset(
        data_path=prompt_data_path,
        max_length=max_length,
        split='train',
        train_split=train_split
    )
    
    val_prompt_dataset = CADPromptDataset(
        data_path=prompt_data_path,
        max_length=max_length,
        split='val',
        train_split=train_split
    )
    
    # Create dataloaders
    train_prompt_loader = DataLoader(
        train_prompt_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_prompt_batch,
        pin_memory=True
    )
    
    val_prompt_loader = DataLoader(
        val_prompt_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_prompt_batch,
        pin_memory=True
    )
    
    # Code dataset for pre-training (optional)
    pretrain_loader = None
    if code_data_path and os.path.exists(code_data_path):
        pretrain_dataset = CADCodeDataset(
            data_path=code_data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            split='train'
        )
        
        pretrain_loader = DataLoader(
            pretrain_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_code_batch,
            pin_memory=True
        )
    
    return train_prompt_loader, val_prompt_loader, pretrain_loader


def save_example_data(base_dir: str = 'data'):
    """Save example prompt and code datasets for testing."""
    os.makedirs(base_dir, exist_ok=True)

    # Example prompts in JSON format (with train/val splits)
    prompts = [
        {
            "prompt": "Generate a cube with side length 2.0",
            "difficulty": "basic",
        },
        {
            "prompt": "Create a sphere with radius 1.5",
            "difficulty": "basic",
        },
        {
            "prompt": "Generate a cylinder with radius 1.0 and height 3.0",
            "difficulty": "basic",
        },
        {
            "prompt": "Create a cone with base radius 1.5 and height 2.5",
            "difficulty": "intermediate",
        },
        {
            "prompt": "Generate a torus with major radius 2.0 and minor radius 0.5",
            "difficulty": "intermediate",
        },
    ]

    prompts_path = os.path.join(base_dir, 'prompts.json')
    prompts_data = {
        'train': prompts,
        'val': prompts,
    }
    with open(prompts_path, 'w', encoding='utf-8') as f:
        json.dump(prompts_data, f, indent=2, ensure_ascii=False)

    # Example CAD code dataset in JSON format (train split only)
    code_examples = [
        {
            "prompt": "Generate a cube",
            "code": """import trimesh\nimport numpy as np\n\n# Create a cube\nvertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])\nfaces = np.array([[0, 1, 2], [0, 2, 3],[4, 5, 6], [4, 6, 7],[0, 1, 5], [0, 5, 4],[2, 3, 7], [2, 7, 6],[0, 3, 7], [0, 7, 4],[1, 2, 6], [1, 6, 5]])\nmesh = trimesh.Trimesh(vertices=vertices, faces=faces)""",
        },
        {
            "prompt": "Create a sphere",
            "code": """import trimesh\n\n# Create a sphere\nsphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)\nmesh = sphere""",
        },
        {
            "prompt": "Generate a cylinder",
            "code": """import trimesh\n\n# Create a cylinder\ncylinder = trimesh.creation.cylinder(radius=1.0, height=2.0, sections=32)\nmesh = cylinder""",
        },
    ]

    code_examples_path = os.path.join(base_dir, 'code_examples.json')
    code_data = {
        'train': code_examples,
    }
    with open(code_examples_path, 'w', encoding='utf-8') as f:
        json.dump(code_data, f, indent=2, ensure_ascii=False)


def print_dataset_stats(data_path: str):
    """Print statistics about the dataset"""
    if not os.path.exists(data_path):
        print(f"Dataset not found: {data_path}")
        return
    
    print("=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    
    if data_path.endswith('.jsonl'):
        prompts = []
        difficulties = {}
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        prompts.append(data)
                        diff = data.get('difficulty', 'unknown')
                        difficulties[diff] = difficulties.get(diff, 0) + 1
                    except:
                        pass
        
        print(f"Total prompts: {len(prompts)}")
        print(f"\nDifficulty distribution:")
        for diff, count in sorted(difficulties.items()):
            print(f"  {diff}: {count} ({count/len(prompts)*100:.1f}%)")
        
        # Show sample prompts
        print(f"\nSample prompts:")
        for i, prompt in enumerate(prompts[:3], 1):
            text = prompt.get('prompt', '')
            diff = prompt.get('difficulty', 'unknown')
            print(f"\n{i}. [{diff}]")
            print(f"   {text[:100]}{'...' if len(text) > 100 else ''}")
    
    print("=" * 80)