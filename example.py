"""
Example usage of the Evil Twins prompt optimization framework.

This script demonstrates how to use the modular framework to run
GCG and soft prompt optimization attacks.
"""

import torch
from pathlib import Path

from evil_twins.config import GCGConfig, SoftPromptConfig, DatasetConfig
from evil_twins.utils import load_model_tokenizer, get_dtype_from_string
from evil_twins.data import DocDataset
from evil_twins.attacks import optim_gcg, optim_soft
from evil_twins.viz import plot_training_curves


def run_gcg_example():
    """Run a GCG optimization example."""
    print("=== Running GCG Example ===")
    
    # Load model and tokenizer
    model, tokenizer = load_model_tokenizer(
        model_name_or_path="EleutherAI/pythia-14m",
        dtype=torch.float32,  # Use float32 for smaller models
        use_flash_attn_2=False
    )
    
    # Create dataset
    dataset = DocDataset(
        model=model,
        tokenizer=tokenizer,
        orig_prompt="Tell me a recipe.",
        optim_prompt="!" * 5,  # Simplified initialization
        n_docs=5,  # Reduced for faster execution
        doc_len=10,  # Reduced for faster execution
        gen_batch_size=2,  # Reduced for memory efficiency
        gen_train_docs=True,
        gen_dev_docs=True
    )
    
    # Configure GCG
    config = GCGConfig(
        n_epochs=5,  # Reduced for faster execution
        kl_every=1,
        batch_size=2,
        top_k=10,  # Reduced for faster execution
        gamma=0.0,
        early_stop_kl=0.0,
        suffix_mode=False
    )
    
    # Run optimization
    results, best_prompt_ids = optim_gcg(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        config=config,
        log_fpath="gcg_example_log.json",
    )
    
    print("GCG optimization complete!")
    print(f"Best prompt text: {tokenizer.decode(best_prompt_ids[0])}")
    
    # Plot results
    try:
        plot_training_curves("gcg_example_log.json", "gcg_example_curves.png")
        print("Training curves saved to gcg_example_curves.png")
    except Exception as e:
        print(f"Could not create training curves: {e}")


def run_soft_prompt_example():
    """Run a soft prompt optimization example."""
    print("\n=== Running Soft Prompt Example ===")
    
    # Load model and tokenizer
    model, tokenizer = load_model_tokenizer(
        model_name_or_path="EleutherAI/pythia-14m",
        dtype=torch.float32,
        use_flash_attn_2=False
    )
    
    # Create dataset
    dataset = DocDataset(
        model=model,
        tokenizer=tokenizer,
        orig_prompt="Write a story about a cat.",
        optim_prompt="!" * 3,  # Shorter for soft prompts
        n_docs=3,  # Reduced for faster execution
        doc_len=8,  # Reduced for faster execution
        gen_batch_size=2,
        gen_train_docs=True,
        gen_dev_docs=True
    )
    
    # Configure soft prompt optimization
    config = SoftPromptConfig(
        n_epochs=3,  # Reduced for faster execution
        kl_every=1,
        batch_size=2,
        learning_rate=1e-3
    )
    
    # Run optimization
    results, best_embs = optim_soft(
        model=model,
        dataset=dataset,
        config=config,
        log_fpath="soft_example_log.json",
        emb_save_fpath="soft_example_embeddings.pt",
    )
    
    print("Soft prompt optimization complete!")
    print(f"Best embeddings shape: {best_embs.shape}")
    
    # Plot results
    try:
        plot_training_curves("soft_example_log.json", "soft_example_curves.png")
        print("Training curves saved to soft_example_curves.png")
    except Exception as e:
        print(f"Could not create training curves: {e}")


def main():
    """Run both examples."""
    print("Evil Twins Prompt Optimization Examples")
    print("=" * 50)
    
    # Create output directory
    Path("./examples").mkdir(exist_ok=True)
    
    try:
        run_gcg_example()
        run_soft_prompt_example()
        print("\nExamples completed successfully!")
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 