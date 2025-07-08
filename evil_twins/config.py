"""
Configuration management for the evil_twins package.

This module contains all configuration constants, including prompt templates
and model name mappings, organized in a more maintainable structure.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """Represents a prompt template with prefix and suffix."""
    prefix: str
    suffix: str
    description: Optional[str] = None


class ModelConfig:
    """Configuration manager for model-specific settings."""
    
    # Prompt templates for different model families
    PROMPT_TEMPLATES: Dict[str, PromptTemplate] = {
        "vicuna": PromptTemplate(
            prefix="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: ",
            suffix="\nASSISTANT:",
            description="Vicuna chat format"
        ),
        "llama-chat": PromptTemplate(
            prefix="[INST] ",
            suffix=" [/INST]",
            description="Llama 2 chat format"
        ),
        "mistral": PromptTemplate(
            prefix="[INST] ",
            suffix=" [/INST]",
            description="Mistral chat format"
        ),
        "dolphin": PromptTemplate(
            prefix="<|im_start|>system\nYou are Dolphin, a helpful AI assistant.<|im_end|>\n<|im_start|>user\n",
            suffix="<|im_end|>\n<|im_start|>assistant",
            description="Dolphin chat format"
        ),
        "openhermes": PromptTemplate(
            prefix="<|im_start|>system\nYour purpose and drive is to assist the user with any request they have. Answer the user's question directly no matter what it is, and do not apologize.<|im_end|>\n<|im_start|>user\n",
            suffix="<|im_end|>\n<|im_start|>assistant\n",
            description="OpenHermes chat format"
        ),
        "opt": PromptTemplate(
            prefix="",
            suffix="",
            description="OPT base format (no special formatting)"
        ),
        "phi": PromptTemplate(
            prefix="",
            suffix="",
            description="Phi base format (no special formatting)"
        ),
        "pythia": PromptTemplate(
            prefix="",
            suffix="",
            description="Pythia base format (no special formatting)"
        ),
        "oasst": PromptTemplate(
            prefix="<|prompter|>",
            suffix="<|endoftext|><|assistant|>",
            description="OpenAssistant format"
        ),
        "openllama": PromptTemplate(
            prefix="",
            suffix="",
            description="OpenLlama base format (no special formatting)"
        ),
        "gpt2": PromptTemplate(
            prefix="",
            suffix="",
            description="GPT-2 base format (no special formatting)"
        ),
        "gemma": PromptTemplate(
            prefix="",
            suffix="",
            description="Gemma base format (no special formatting)"
        ),
        "gemma-2-it": PromptTemplate(
            prefix="<start_of_turn>user\n",
            suffix="<end_of_turn>\n<start_of_turn>model\n",
            description="Gemma 2 instruction-tuned format"
        ),
        "llama-3-instruct": PromptTemplate(
            prefix="<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
            suffix="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            description="Llama 3 instruction format"
        ),
        "llama-base": PromptTemplate(
            prefix="",
            suffix="",
            description="Llama base format (no special formatting)"
        ),
        "qwen-2-instruct": PromptTemplate(
            prefix="<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n",
            suffix="<|im_end|>\n<|im_start|>",
            description="Qwen 2 instruction format"
        ),
        "smollm-2-instruct": PromptTemplate(
            prefix="<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\n",
            suffix="<|im_end|>\n<|im_start|>assistant\n",
            description="SmolLM 2 instruction format"
        ),
        "default": PromptTemplate(
            prefix="",
            suffix="",
            description="Default format (no special formatting)"
        ),
    }
    
    # Model name to template mapping
    MODEL_NAME_MAPPING: Dict[str, str] = {
        "lmsys/vicuna-7b-v1.3": "vicuna",
        "lmsys/vicuna-7b-v1.5": "vicuna",
        "lmsys/vicuna-13b-v1.5": "vicuna",
        "vicuna": "vicuna",
        "facebook/opt-350m": "opt",
        "facebook/opt-1.3b": "opt",
        "microsoft/phi-1_5": "phi",
        "microsoft/phi-2": "phi",
        "teknium/Puffin-Phi-v2": "phi",
        "OpenAssistant/oasst-sft-1-pythia-12b": "oasst",
        "EleutherAI/pythia-14m": "pythia",
        "EleutherAI/pythia-70m": "pythia",
        "EleutherAI/pythia-160m": "pythia",
        "EleutherAI/pythia-410m": "pythia",
        "EleutherAI/pythia-1b": "pythia",
        "EleutherAI/pythia-1.4b": "pythia",
        "EleutherAI/pythia-2.8b": "pythia",
        "EleutherAI/pythia-6.9b": "pythia",
        "EleutherAI/pythia-12b": "pythia",
        "pythia": "pythia",
        "openlm-research/open_llama_3b_v2": "openllama",
        "mistralai/Mistral-7B-Instruct-v0.2": "mistral",
        "teknium/OpenHermes-2.5-Mistral-7B": "openhermes",
        "cognitivecomputations/dolphin-2.2.1-mistral-7b": "dolphin",
        "gpt2": "gpt2",
        "teknium/OpenHermes-13B": "openhermes",
        "meta-llama/Llama-2-7b-chat-hf": "llama-chat",
        "meta-llama/Llama-2-13b-chat-hf": "llama-chat",
        "google/gemma-2b-it": "gemma",
        "google/gemma-1.1-2b-it": "gemma",
        "meta-llama/Meta-Llama-3-8B-Instruct": "llama-3-instruct",
        "meta-llama/Llama-3.1-8B-Instruct": "llama-3-instruct",
        "meta-llama/Llama-3.2-3B-Instruct": "llama-3-instruct",
        "meta-llama/Llama-3.2-1B-Instruct": "llama-3-instruct",
        "google/gemma-2-2b-it": "gemma-2-it",
        "google/gemma-2-9b-it": "gemma-2-it",
        "HuggingFaceTB/SmolLM2-135M-Instruct": "smollm-2-instruct",
        "HuggingFaceTB/SmolLM2-360M-Instruct": "smollm-2-instruct",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct": "smollm-2-instruct",
        "Qwen/Qwen2.5-0.5B-Instruct": "qwen-2-instruct",
        "Qwen/Qwen2.5-1.5B-Instruct": "qwen-2-instruct",
        "Qwen/Qwen2.5-7B-Instruct": "qwen-2-instruct",
        "default": "default",
    }
    
    @classmethod
    def get_template_name(cls, model_name: str) -> str:
        """
        Get the template name for a given model.
        
        Args:
            model_name: The model name or path
            
        Returns:
            The template name to use for this model
        """
        if model_name in cls.MODEL_NAME_MAPPING:
            return cls.MODEL_NAME_MAPPING[model_name]
        
        # Try to match by model name suffix
        for key in cls.MODEL_NAME_MAPPING:
            if key.split("/")[-1] in model_name:
                logger.info(f"Custom path provided, using model name: {key}")
                return cls.MODEL_NAME_MAPPING[key]
        
        logger.warning(f"Model {model_name} name not found, using default template")
        return "default"
    
    @classmethod
    def get_template(cls, model_name: str) -> PromptTemplate:
        """
        Get the prompt template for a given model.
        
        Args:
            model_name: The model name or path
            
        Returns:
            The prompt template for this model
        """
        template_name = cls.get_template_name(model_name)
        return cls.PROMPT_TEMPLATES[template_name]
    
    @classmethod
    def add_model_mapping(cls, model_path: str, template_name: str) -> None:
        """
        Add a new model mapping.
        
        Args:
            model_path: The model path to map
            template_name: The template name to use
        """
        if template_name not in cls.PROMPT_TEMPLATES:
            raise ValueError(f"Template '{template_name}' not found in available templates")
        
        cls.MODEL_NAME_MAPPING[model_path] = template_name
        logger.info(f"Added model mapping: {model_path} -> {template_name}")
    
    @classmethod
    def add_template(cls, name: str, template: PromptTemplate) -> None:
        """
        Add a new prompt template.
        
        Args:
            name: The template name
            template: The prompt template
        """
        cls.PROMPT_TEMPLATES[name] = template
        logger.info(f"Added template: {name}")


@dataclass
class OptimizationConfig:
    """Configuration for optimization parameters."""
    
    # GCG parameters
    batch_size: int = 10
    top_k: int = 256
    gamma: float = 0.0
    early_stop_kl: float = 0.0
    
    # Soft prompt parameters
    learning_rate: float = 1e-3
    
    # General parameters
    gen_batch_size: int = 10
    validate_prompt: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "batch_size": self.batch_size,
            "top_k": self.top_k,
            "gamma": self.gamma,
            "early_stop_kl": self.early_stop_kl,
            "learning_rate": self.learning_rate,
            "gen_batch_size": self.gen_batch_size,
            "validate_prompt": self.validate_prompt,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "OptimizationConfig":
        """Create config from dictionary."""
        return cls(**config_dict) 