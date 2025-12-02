import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Optional, Tuple
import torch.nn.functional as F

SYSTEM_PROMPT = (
    "You are a CAD code generation assistant. "
    "First, reason about the problem inside <think>...</think> tags. "
    "Then, output only a valid CADQuery Python script inside <script>...</script> tags. "
    "The script must be complete, executable CADQuery code that creates the 3D design described by the user. "
    "Do not include any explanations, comments, or text outside these tags."
)


def extract_script_from_text(text: str) -> str:
    """Extract the content inside <script>...</script> tags from a string.

    Returns an empty string if the tags are missing or malformed.
    """
    start_tag = "<script>"
    end_tag = "</script>"

    start_idx = text.find(start_tag)
    end_idx = text.find(end_tag, start_idx + len(start_tag)) if start_idx != -1 else -1

    if start_idx == -1 or end_idx == -1:
        return ""

    return text[start_idx + len(start_tag) : end_idx].strip()


class CADGeneratorModel(nn.Module):
    """
    CAD Code Generator Model based on pretrained language model
    Can be any decoder-only model (GPT-2, CodeGen, etc.)
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-7B",
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1
    ):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None
        )
        
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
                target_modules=["c_attn", "c_proj"],  # For GPT-2
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
        max_length: int = 512,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        num_return_sequences: int = 1
    ) -> torch.Tensor:
        """Generate CAD code"""
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
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
    def __init__(self, hidden_size: int):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
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
        model_name: str = "Qwen/Qwen2.5-Coder-7B",
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1
    ):
        super().__init__()
        
        self.generator = CADGeneratorModel(
            model_name=model_name,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        
        self.value_head = ValueHead(self.generator.hidden_size)
        
        self.tokenizer = self.generator.tokenizer
    
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
        max_length: int = 512,
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
            max_length=max_length,
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
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-7B"):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
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