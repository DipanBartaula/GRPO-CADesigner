import torch
import argparse
import os
from typing import List, Optional
import trimesh

from model import PPOCADModel
from utils import cad_code_to_mesh, render_mesh
import matplotlib.pyplot as plt
import numpy as np

class CADInference:
    """Inference class for CAD generation"""
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_name: str = 'gpt2',
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = PPOCADModel(model_name=model_name)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("No checkpoint provided. Using pretrained base model.")
        
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = self.model.tokenizer
    
    def generate(
        self,
        prompts: List[str],
        max_length: int = 512,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        Generate CAD code from prompts
        
        Args:
            prompts: List of text prompts
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            num_return_sequences: Number of sequences to generate per prompt
        
        Returns:
            generated_codes: List of generated CAD codes
        """
        # Tokenize prompts
        encodings = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences
            )
        
        # Decode
        generated_codes = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        return generated_codes
    
    def generate_and_render(
        self,
        prompt: str,
        output_dir: str = 'outputs',
        **generation_kwargs
    ) -> tuple[str, Optional[trimesh.Trimesh]]:
        """
        Generate CAD code and render the result
        
        Args:
            prompt: Text prompt
            output_dir: Directory to save outputs
            **generation_kwargs: Additional generation parameters
        
        Returns:
            code: Generated CAD code
            mesh: Generated mesh (None if execution failed)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate code
        codes = self.generate([prompt], **generation_kwargs)
        code = codes[0]
        
        # Save code
        code_path = os.path.join(output_dir, 'generated_code.py')
        with open(code_path, 'w') as f:
            f.write(code)
        print(f"Saved code to {code_path}")
        
        # Execute code to get mesh
        mesh = cad_code_to_mesh(code)
        
        if mesh is not None:
            # Save mesh
            mesh_path = os.path.join(output_dir, 'generated_mesh.obj')
            mesh.export(mesh_path)
            print(f"Saved mesh to {mesh_path}")
            
            # Render views
            views = render_mesh(mesh, views=4)
            
            # Save rendered views
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            for idx, (ax, view) in enumerate(zip(axes.flat, views)):
                ax.imshow(view)
                ax.axis('off')
                ax.set_title(f'View {idx+1}')
            
            plt.tight_layout()
            render_path = os.path.join(output_dir, 'rendered_views.png')
            plt.savefig(render_path)
            plt.close()
            print(f"Saved rendered views to {render_path}")
        else:
            print("Failed to execute code or generate mesh")
        
        return code, mesh
    
    def batch_generate(
        self,
        prompts: List[str],
        output_dir: str = 'outputs',
        **generation_kwargs
    ) -> List[tuple[str, Optional[trimesh.Trimesh]]]:
        """
        Generate CAD code for multiple prompts
        
        Args:
            prompts: List of text prompts
            output_dir: Directory to save outputs
            **generation_kwargs: Additional generation parameters
        
        Returns:
            results: List of (code, mesh) tuples
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            print(f"\nGenerating for prompt {i+1}/{len(prompts)}: {prompt}")
            prompt_output_dir = os.path.join(output_dir, f'prompt_{i}')
            
            code, mesh = self.generate_and_render(
                prompt,
                output_dir=prompt_output_dir,
                **generation_kwargs
            )
            
            results.append((code, mesh))
        
        return results


def main():
    parser = argparse.ArgumentParser(description='CAD Generation Inference')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--prompt', type=str, required=True, help='Generation prompt')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--max_length', type=int, default=512, help='Max generation length')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.95, help='Nucleus sampling')
    parser.add_argument('--model_name', type=str, default='gpt2', help='Base model name')
    
    args = parser.parse_args()
    
    # Initialize inference
    inferencer = CADInference(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name
    )
    
    # Generate and render
    print(f"Generating CAD code for prompt: {args.prompt}")
    code, mesh = inferencer.generate_and_render(
        prompt=args.prompt,
        output_dir=args.output_dir,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    
    print("\nGenerated Code:")
    print("-" * 80)
    print(code)
    print("-" * 80)
    
    if mesh is not None:
        print(f"\nMesh Statistics:")
        print(f"  Vertices: {len(mesh.vertices)}")
        print(f"  Faces: {len(mesh.faces)}")
        print(f"  Watertight: {mesh.is_watertight}")
        print(f"  Volume: {mesh.volume if mesh.is_watertight else 'N/A'}")
    else:
        print("\nFailed to generate valid mesh")


if __name__ == '__main__':
    main()