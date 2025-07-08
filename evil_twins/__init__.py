"""
Evil Twins: Prompt Optimization Framework

A modular framework for optimizing prompts using various attack methods including
GCG (Gradient-based optimization) and soft prompts.
"""

__version__ = "0.1.0"

from .config import (
    GCGConfig,
    SoftPromptConfig,
    DatasetConfig,
    ModelConfig,
    PROMPT_TEMPLATES,
    MODEL_NAME_OR_PATH_TO_NAME,
)
from .utils import load_model_tokenizer, build_prompt, extract_prompt_from_template
from .data import DocDataset
from .attacks import optim_gcg, optim_soft
from .evaluation import compute_dataset_kl, compute_perplexity
from .viz import plot_training_curves, plot_prompt_comparison, plot_kl_convergence

__all__ = [
    # Config classes
    "GCGConfig",
    "SoftPromptConfig", 
    "DatasetConfig",
    "ModelConfig",
    "PROMPT_TEMPLATES",
    "MODEL_NAME_OR_PATH_TO_NAME",
    
    # Utility functions
    "load_model_tokenizer",
    "build_prompt",
    "extract_prompt_from_template",
    
    # Data classes
    "DocDataset",
    
    # Attack functions
    "optim_gcg",
    "optim_soft",
    
    # Evaluation functions
    "compute_dataset_kl",
    "compute_perplexity",
    
    # Visualization functions
    "plot_training_curves",
    "plot_prompt_comparison", 
    "plot_kl_convergence",
] 