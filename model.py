import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Optional, Tuple
import torch.nn.functional as F
import re
import os

# Comprehensive system prompt - generates ONLY code inside <script> tags
SYSTEM_PROMPT = '''You are a CadQuery Python code generator. Generate ONLY executable Python code.

OUTPUT FORMAT - You must output EXACTLY this format:
<script>
import cadquery as cq
[your code here]
result = [final CadQuery object]
</script>

ABSOLUTE RULES:
1. Output ONLY the <script> tags with Python code inside - NOTHING ELSE
2. NO thinking, NO explanations, NO comments outside code
3. NO markdown formatting (no ```, no #headers, no **bold**)
4. NO box-drawing characters (─│┌┐└┘├┤┬┴┼)
5. NO tables, diagrams, ASCII art, or decorations
6. NO text before <script> or after </script>
7. Code MUST be syntactically valid Python
8. Code MUST start with: import cadquery as cq
9. Code MUST end with result variable containing the 3D object

CADQUERY REFERENCE:

Basic Shapes:
  cq.Workplane("XY").box(length, width, height)     # Rectangular box
  cq.Workplane("XY").cylinder(height, radius)       # Cylinder
  cq.Workplane("XY").sphere(radius)                 # Sphere
  cq.Workplane("XY").cone(height, r1, r2)           # Cone/truncated cone

2D to 3D Operations:
  .rect(width, height).extrude(depth)               # Extruded rectangle
  .circle(radius).extrude(depth)                    # Extruded circle
  .polygon(n_sides, radius).extrude(depth)          # Extruded polygon
  .ellipse(x_radius, y_radius).extrude(depth)       # Extruded ellipse
  .rect(w, h).revolve(angle)                        # Revolved rectangle

Modifications:
  .faces(">Z").hole(diameter, depth)                # Hole through face
  .edges().fillet(radius)                           # Round edges
  .edges().chamfer(length)                          # Bevel edges
  .shell(thickness)                                 # Hollow out solid

Boolean Operations:
  .cut(other_shape)                                 # Subtract shape
  .union(other_shape)                               # Add shape
  .intersect(other_shape)                           # Intersection

Positioning:
  .translate((x, y, z))                             # Move object
  .rotate((0,0,0), (0,0,1), angle)                  # Rotate object
  .mirror("XY")                                     # Mirror object

Face Selection:
  .faces(">Z")   # Top face       .faces("<Z")   # Bottom face
  .faces(">X")   # Right face     .faces("<X")   # Left face
  .faces(">Y")   # Front face     .faces("<Y")   # Back face

EXAMPLES:

Example 1 - Simple cube:
<script>
import cadquery as cq
result = cq.Workplane("XY").box(10, 10, 10)
</script>

Example 2 - Cylinder with hole:
<script>
import cadquery as cq
result = cq.Workplane("XY").cylinder(20, 10).faces(">Z").hole(5)
</script>

Example 3 - Box with fillet edges:
<script>
import cadquery as cq
result = cq.Workplane("XY").box(30, 20, 10).edges().fillet(2)
</script>

Example 4 - L-bracket:
<script>
import cadquery as cq
result = (cq.Workplane("XY")
    .box(20, 10, 5)
    .faces(">Y")
    .workplane()
    .box(10, 5, 15))
</script>

Example 5 - Hollow box:
<script>
import cadquery as cq
result = cq.Workplane("XY").box(20, 20, 20).faces(">Z").shell(-2)
</script>

Now generate CadQuery code for the user's request. Output ONLY <script> tags with valid Python code inside.'''


def sanitize_code(code: str) -> str:
    """
    Sanitize generated code by removing invalid characters.
    
    Removes box-drawing characters, unicode replacement chars, and other
    non-ASCII characters that aren't valid in Python.
    """
    if not code:
        return code
    
    # Remove box-drawing characters (U+2500-U+257F): ┌ ─ ┐ │ └ ┘ etc.
    code = re.sub(r'[\u2500-\u257F]', '', code)
    
    # Remove unicode replacement character
    code = code.replace('\ufffd', '')
    
    # Remove other problematic unicode - keep only ASCII
    cleaned_chars = []
    for char in code:
        if ord(char) < 128:  # ASCII only
            cleaned_chars.append(char)
        elif ord(char) in [0x201C, 0x201D]:  # Fancy double quotes
            cleaned_chars.append('"')
        elif ord(char) in [0x2018, 0x2019]:  # Fancy single quotes
            cleaned_chars.append("'")
        # Skip all other unicode
    
    code = ''.join(cleaned_chars)
    
    # Remove lines that are just separators or box drawings
    lines = code.split('\n')
    valid_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip empty lines and separator-only lines
        if stripped and not re.match(r'^[\-=_+|/\\*#<>]+$', stripped):
            valid_lines.append(line)
    
    return '\n'.join(valid_lines)


def extract_script_from_text(text: str) -> str:
    """Extract the content inside <script>...</script> tags from a string.

    Returns an empty string if the tags are missing or malformed.
    Also sanitizes the extracted code to remove invalid characters.
    """
    start_tag = "<script>"
    end_tag = "</script>"

    start_idx = text.find(start_tag)
    end_idx = text.find(end_tag, start_idx + len(start_tag)) if start_idx != -1 else -1

    if start_idx == -1 or end_idx == -1:
        # Try to find any Python-like code as fallback
        # Look for 'import cadquery' pattern
        if 'import cadquery' in text:
            # Extract from 'import cadquery' to end or next obvious boundary
            import_idx = text.find('import cadquery')
            code_candidate = text[import_idx:]
            # Cut at common boundaries
            for boundary in ['</script>', '</think>', '\n\n\n', '```']:
                if boundary in code_candidate:
                    code_candidate = code_candidate[:code_candidate.find(boundary)]
            return sanitize_code(code_candidate.strip())
        return ""

    script = text[start_idx + len(start_tag) : end_idx].strip()
    return sanitize_code(script)


class CADGeneratorModel(nn.Module):
    """
    CAD Code Generator Model based on pretrained language model
    Can be any decoder-only model (GPT-2, CodeGen, etc.)
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",  # Instruct variant for better following instructions
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        use_gradient_checkpointing: bool = True,  # Memory optimization
        load_in_8bit: bool = False,  # Quantization for memory savings
        load_in_4bit: bool = False,  # More aggressive quantization
    ):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Decoder-only model: use left padding for correct generation behavior
        self.tokenizer.padding_side = "left"
        
        # Configure model loading based on memory optimization settings
        load_kwargs = {
            "torch_dtype": torch.float16 if (load_in_8bit or load_in_4bit) else torch.float32,
            "device_map": "auto" if (load_in_8bit or load_in_4bit) else None,
        }
        
        # Add quantization config if requested
        if load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                print(f"[CADGeneratorModel] Loading model in 8-bit quantization")
            except ImportError:
                print(f"[CADGeneratorModel] Warning: bitsandbytes not installed, loading in FP32")
        elif load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                print(f"[CADGeneratorModel] Loading model in 4-bit quantization")
            except ImportError:
                print(f"[CADGeneratorModel] Warning: bitsandbytes not installed, loading in FP32")
        
        print(f"[CADGeneratorModel] Loading {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        
        # Enable gradient checkpointing for memory optimization
        if use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print(f"[CADGeneratorModel] Gradient checkpointing enabled")
        
        # Apply LoRA if enabled
        if use_lora:
            self.apply_lora(lora_r, lora_alpha, lora_dropout)
        
        self.vocab_size = self.model.config.vocab_size
        self.hidden_size = self.model.config.hidden_size
    
    def apply_lora(self, r: int, alpha: int, dropout: float):
        """Apply LoRA to the model"""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            lora_config = LoraConfig(
                r=r,
                lora_alpha=alpha,
                target_modules=["W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            self.model = get_peft_model(self.model, lora_config)
            print("LoRA applied successfully")
        except Exception as e:
            print(f"Could not apply LoRA: {e}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        
        return {
            'logits': outputs.logits,
            'loss': outputs.loss if labels is not None else None,
            'hidden_states': outputs.hidden_states[-1]
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 1024,  # max_new_tokens (reduced for faster generation)
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        num_return_sequences: int = 1
    ) -> torch.Tensor:
        """Generate CAD code"""
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_length,  # Use max_new_tokens to avoid warning
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        return outputs
    
    def get_trainable_parameters(self):
        """Get number of trainable parameters"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


class ValueHead(nn.Module):
    """Value head for PPO"""
    def __init__(self, hidden_size: int, dtype=torch.float32):  
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size).to(dtype),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1).to(dtype)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Take the last token's hidden state
        last_hidden = hidden_states[:, -1, :]
        value = self.value_head(last_hidden)
        return value.squeeze(-1)


class PPOCADModel(nn.Module):
    """PPO Model with Policy and Value heads"""
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",  # Instruct variant for better instruction following
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        use_gradient_checkpointing: bool = True,  # Memory optimization
        load_in_8bit: bool = False,  # Quantization for memory savings
        load_in_4bit: bool = False,  # More aggressive quantization
    ):
        super().__init__()
        
        self.generator = CADGeneratorModel(
            model_name=model_name,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_gradient_checkpointing=use_gradient_checkpointing,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
        )
        
        self.value_head = ValueHead(self.generator.hidden_size, dtype=torch.float32)
        
        self.tokenizer = self.generator.tokenizer

        # Freeze base model weights when using LoRA; only LoRA and value head remain trainable
        for name, param in self.generator.model.named_parameters():
            # LoRA parameters in peft models typically contain "lora_" in their names
            param.requires_grad = ("lora_" in name)

        for param in self.value_head.parameters():
            param.requires_grad = True
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass returning both policy and value"""
        outputs = self.generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        values = self.value_head(outputs['hidden_states'])
        
        return {
            'logits': outputs['logits'],
            'values': values,
            'loss': outputs['loss'],
            'hidden_states': outputs['hidden_states']
        }
    
    def generate(self, *args, **kwargs):
        """Generate CAD code"""
        return self.generator.generate(*args, **kwargs)
    
    def generate_cad_script_with_log_probs(
        self,
        design_prompt: str,
        max_length: int = 32768,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95
    ) -> Tuple[str, torch.Tensor]:
        prompt = (
            SYSTEM_PROMPT
            + "\nUser design description:\n"
            + design_prompt
            + "\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.generator.model.device)
        attention_mask = inputs["attention_mask"].to(self.generator.model.device)

        generated_ids = self.generator.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        full_text = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )

        start_tag = "<script>"
        end_tag = "</script>"
        start_idx = full_text.find(start_tag)
        end_idx = full_text.find(end_tag, start_idx + len(start_tag))

        if start_idx == -1 or end_idx == -1:
            script = ""
        else:
            script = full_text[start_idx + len(start_tag) : end_idx].strip()

        script_prompt = prompt + script
        script_inputs = self.tokenizer(script_prompt, return_tensors="pt")
        script_input_ids = script_inputs["input_ids"].to(self.generator.model.device)
        script_attention_mask = script_inputs["attention_mask"].to(
            self.generator.model.device
        )

        all_log_probs, _ = self.get_log_probs(
            script_input_ids, script_attention_mask
        )

        script_only_ids = self.tokenizer(script, return_tensors="pt")["input_ids"][
            0
        ]
        script_len = script_only_ids.shape[0]
        script_log_probs = all_log_probs[:, -script_len:]

        return script, script_log_probs
    
    def get_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get log probabilities and values for generated sequences"""
        outputs = self.forward(input_ids, attention_mask)
        
        logits = outputs['logits']
        values = outputs['values']
        
        # Get log probs
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Get log prob of actual tokens
        selected_log_probs = torch.gather(
            log_probs[:, :-1, :],
            dim=2,
            index=input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask padding
        if attention_mask is not None:
            selected_log_probs = selected_log_probs * attention_mask[:, 1:]
        
        return selected_log_probs, values
    
    def save_pretrained(self, save_directory: str):
        """Save model"""
        self.generator.model.save_pretrained(save_directory)
        self.generator.tokenizer.save_pretrained(save_directory)
        
        # Save value head
        torch.save(self.value_head.state_dict(), f"{save_directory}/value_head.pt")
    
    def load_pretrained(self, load_directory: str):
        """Load model"""
        self.generator.model = AutoModelForCausalLM.from_pretrained(load_directory)
        self.generator.tokenizer = AutoTokenizer.from_pretrained(load_directory)
        
        # Load value head
        value_head_path = f"{load_directory}/value_head.pt"
        if os.path.exists(value_head_path):
            self.value_head.load_state_dict(torch.load(value_head_path))


class ReferenceModel(nn.Module):
    """Reference model for KL penalty in PPO"""
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"):  # Instruct variant for better instruction following
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Load Qwen model in FP32
            device_map=None
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Ensure reference tokenizer also uses left padding
        self.tokenizer.padding_side = "left"
        
        # Freeze reference model
        for param in self.model.parameters():
            param.requires_grad = False
    
    def get_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get log probabilities from reference model"""
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_probs = torch.gather(
                log_probs[:, :-1, :],
                dim=2,
                index=input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            
            if attention_mask is not None:
                selected_log_probs = selected_log_probs * attention_mask[:, 1:]
            
            return selected_log_probs