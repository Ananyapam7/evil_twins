"""
Main command-line interface for prompt optimization.
"""

import argparse
import os
import sys
from pathlib import Path
import torch

from .config import (
    GCGConfig, 
    SoftPromptConfig, 
    DatasetConfig, 
    ModelConfig,
    DEFAULT_BATCH_SIZE,
    DEFAULT_TOP_K,
    DEFAULT_GAMMA,
    DEFAULT_LEARNING_RATE,
    DEFAULT_N_EPOCHS,
    DEFAULT_KL_EVERY,
    DEFAULT_EARLY_STOP_KL,
    DEFAULT_N_DOCS,
    DEFAULT_DOC_LEN,
    DEFAULT_GEN_BATCH_SIZE,
)
from .utils import load_model_tokenizer, get_dtype_from_string
from .data import DocDataset
from .attacks import optim_gcg, optim_soft
from .viz import plot_training_curves


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prompt optimization using GCG or soft prompts"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="EleutherAI/pythia-14m",
        help="Model name or path to load"
    )
    parser.add_argument(
        "--dtype", 
        type=str, 
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype"
    )
    parser.add_argument(
        "--device_map", 
        type=str, 
        default="auto",
        help="Device map for model loading"
    )
    parser.add_argument(
        "--use_flash_attn_2", 
        action="store_true",
        help="Use flash attention 2"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--orig_prompt", 
        type=str, 
        default="Tell me a recipe.",
        help="Original prompt to optimize"
    )
    parser.add_argument(
        "--optim_prompt", 
        type=str, 
        default="!" * 5,
        help="Initial optimized prompt"
    )
    parser.add_argument(
        "--n_docs", 
        type=int, 
        default=DEFAULT_N_DOCS,
        help="Number of documents to generate"
    )
    parser.add_argument(
        "--doc_len", 
        type=int, 
        default=DEFAULT_DOC_LEN,
        help="Length of each document"
    )
    parser.add_argument(
        "--gen_batch_size", 
        type=int, 
        default=DEFAULT_GEN_BATCH_SIZE,
        help="Batch size for document generation"
    )
    
    # Attack type
    parser.add_argument(
        "--attack_type", 
        type=str, 
        default="gcg",
        choices=["gcg", "soft"],
        help="Type of attack to run"
    )
    
    # GCG specific arguments
    parser.add_argument(
        "--n_epochs", 
        type=int, 
        default=DEFAULT_N_EPOCHS,
        help="Number of optimization epochs"
    )
    parser.add_argument(
        "--kl_every", 
        type=int, 
        default=DEFAULT_KL_EVERY,
        help="Compute KL divergence every N epochs"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for optimization"
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=DEFAULT_TOP_K,
        help="Top-k tokens to consider for replacement (GCG only)"
    )
    parser.add_argument(
        "--gamma", 
        type=float, 
        default=DEFAULT_GAMMA,
        help="Fluency penalty coefficient"
    )
    parser.add_argument(
        "--early_stop_kl", 
        type=float, 
        default=DEFAULT_EARLY_STOP_KL,
        help="Early stopping KL threshold"
    )
    parser.add_argument(
        "--suffix_mode", 
        action="store_true",
        help="Run in suffix mode (GCG only)"
    )
    
    # Soft prompt specific arguments
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate for soft prompt optimization"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--log_fpath", 
        type=str, 
        default=None,
        help="Path for log file (auto-generated if not specified)"
    )
    parser.add_argument(
        "--emb_save_fpath", 
        type=str, 
        default=None,
        help="Path for embedding save file (soft prompt only)"
    )
    parser.add_argument(
        "--no_plot", 
        action="store_true",
        help="Skip plotting training curves"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up file paths
    if args.log_fpath is None:
        args.log_fpath = output_dir / f"{args.attack_type}_optimization_log.json"
    
    if args.attack_type == "soft" and args.emb_save_fpath is None:
        args.emb_save_fpath = output_dir / f"{args.attack_type}_embeddings.pt"
    
    print(f"Loading model: {args.model_name}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_tokenizer(
        model_name_or_path=args.model_name,
        dtype=get_dtype_from_string(args.dtype),
        device_map=args.device_map,
        use_flash_attn_2=args.use_flash_attn_2,
        eval_mode=True,
    )
    
    print(f"Creating dataset...")
    
    # Create dataset
    dataset = DocDataset(
        model=model,
        tokenizer=tokenizer,
        orig_prompt=args.orig_prompt,
        optim_prompt=args.optim_prompt,
        n_docs=args.n_docs,
        doc_len=args.doc_len,
        gen_batch_size=args.gen_batch_size,
        validate_prompt=True,
        gen_train_docs=True,
        gen_dev_docs=True,
    )
    
    print(f"Starting {args.attack_type.upper()} optimization...")
    
    # Run optimization
    if args.attack_type == "gcg":
        config = GCGConfig(
            n_epochs=args.n_epochs,
            kl_every=args.kl_every,
            batch_size=args.batch_size,
            top_k=args.top_k,
            gamma=args.gamma,
            early_stop_kl=args.early_stop_kl,
            suffix_mode=args.suffix_mode,
        )
        
        results, best_prompt_ids = optim_gcg(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            config=config,
            log_fpath=str(args.log_fpath),
        )
        
        print("GCG optimization complete!")
        print(f"Best prompt IDs: {best_prompt_ids}")
        print(f"Best prompt text: {tokenizer.decode(best_prompt_ids[0])}")
        
    elif args.attack_type == "soft":
        config = SoftPromptConfig(
            n_epochs=args.n_epochs,
            kl_every=args.kl_every,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        
        results, best_embs = optim_soft(
            model=model,
            dataset=dataset,
            config=config,
            log_fpath=str(args.log_fpath),
            emb_save_fpath=str(args.emb_save_fpath),
        )
        
        print("Soft prompt optimization complete!")
        print(f"Best embeddings saved to: {args.emb_save_fpath}")
    
    # Plot training curves
    if not args.no_plot:
        try:
            plot_fpath = output_dir / f"{args.attack_type}_training_curves.png"
            plot_training_curves(str(args.log_fpath), str(plot_fpath))
            print(f"Training curves saved to: {plot_fpath}")
        except Exception as e:
            print(f"Could not create training curves: {e}")
    
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main() 