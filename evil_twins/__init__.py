"""
Evil Twins: Prompt Optimization Package

This package provides tools for optimizing prompts using both hard (GCG) and soft
prompt optimization techniques. It implements the methods described in the paper
"Prompts have evil twins" (EMNLP 2024).

Main components:
- ModelManager: For loading and managing models and tokenizers
- DocDataset: For generating and managing document datasets
- GCGOptimizer: For hard prompt optimization using GCG
- SoftPromptOptimizer: For soft prompt optimization
- TrainingVisualizer: For visualizing training results

Example usage:
    from evil_twins import ModelManager, DocDataset, GCGOptimizer, OptimizationConfig
    
    # Load model and tokenizer
    model_manager = ModelManager("gpt2")
    model, tokenizer = model_manager.load_model_and_tokenizer()
    
    # Create dataset
    dataset = DocDataset(
        model=model,
        tokenizer=tokenizer,
        orig_prompt="Tell me a recipe.",
        optim_prompt="! " * 5,
        n_docs=10,
        doc_len=16,
    )
    
    # Configure optimization
    config = OptimizationConfig(
        batch_size=4,
        top_k=64,
        gamma=0.0,
    )
    
    # Run optimization
    optimizer = GCGOptimizer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        config=config,
        log_fpath="optimization_log.json",
    )
    
    results, best_prompt = optimizer.optimize(n_epochs=10)
"""

# Import main classes and functions
from .models import ModelManager, load_model_tokenizer
from .datasets import DocDataset
from .optimizers import GCGOptimizer, SoftPromptOptimizer, BaseOptimizer
from .config import OptimizationConfig, ModelConfig
from .prompts import PromptBuilder, build_prompt, extract_prompt_from_template
from .visualization import TrainingVisualizer, plot_training_curves, create_comprehensive_visualization

# Import legacy functions for backward compatibility
from .optimizers.gcg import optim_gcg
from .optimizers.soft import optim_soft

# Import utility functions
from .utils import (
    compute_neg_log_prob,
    compute_grads,
    replace_tok,
    compute_dataset_kl,
    OrigModelEmbs,
)

# Import soft prompt embedding layer
from .optimizers.soft import SoftPromptEmbeddingLayer

__version__ = "0.0.1"

__all__ = [
    # Main classes
    "ModelManager",
    "DocDataset",
    "GCGOptimizer",
    "SoftPromptOptimizer",
    "BaseOptimizer",
    "TrainingVisualizer",
    
    # Configuration
    "OptimizationConfig",
    "ModelConfig",
    
    # Prompt management
    "PromptBuilder",
    "build_prompt",
    "extract_prompt_from_template",
    
    # Model management
    "load_model_tokenizer",
    
    # Optimization (legacy)
    "optim_gcg",
    "optim_soft",
    
    # Utilities
    "compute_neg_log_prob",
    "compute_grads",
    "replace_tok",
    "compute_dataset_kl",
    "OrigModelEmbs",
    "SoftPromptEmbeddingLayer",
    
    # Visualization
    "plot_training_curves",
    "create_comprehensive_visualization",
]
