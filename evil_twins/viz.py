"""
Visualization functions for prompt optimization results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Dict


def plot_training_curves(log_fpath: str, save_fpath: str = "training_curves.png"):
    """
    Plot training curves from the log file and save the plot.
    
    Args:
        log_fpath: Path to the JSON log file
        save_fpath: Path to save the plot image
    """
    # Load the log data
    with open(log_fpath, 'r') as f:
        log_data = json.load(f)
    
    # Extract data for plotting
    epochs = [entry['epoch'] for entry in log_data]
    losses = [entry['loss'] for entry in log_data]
    best_losses = [entry.get('best_loss', losses[i]) for i, entry in enumerate(log_data)]
    best_kls = [entry.get('best_kl', 0) for entry in log_data]
    cur_kls = [entry.get('cur_kl', 0) for entry in log_data]
    nll_prompts = [entry.get('nll_prompt', 0) for entry in log_data]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Loss curves
    axes[0, 0].plot(epochs, losses, 'b-', label='Current Loss', linewidth=2)
    axes[0, 0].plot(epochs, best_losses, 'r--', label='Best Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: KL divergence curves
    axes[0, 1].plot(epochs, best_kls, 'g-', label='Best KL', linewidth=2)
    axes[0, 1].plot(epochs, cur_kls, 'orange', label='Current KL', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('KL Divergence')
    axes[0, 1].set_title('KL Divergence')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: NLL Prompt curve
    axes[1, 0].plot(epochs, nll_prompts, 'purple', label='NLL Prompt', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Negative Log Likelihood')
    axes[1, 0].set_title('Prompt Negative Log Likelihood')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Combined view
    ax2 = axes[1, 1].twinx()
    line1 = axes[1, 1].plot(epochs, losses, 'b-', label='Loss', linewidth=2)
    line2 = ax2.plot(epochs, cur_kls, 'r-', label='KL', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss', color='b')
    ax2.set_ylabel('KL Divergence', color='r')
    axes[1, 1].set_title('Loss vs KL Divergence')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    axes[1, 1].legend(lines, labels, loc='upper right')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_fpath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {save_fpath}")


def plot_prompt_comparison(
    log_fpath: str, 
    save_fpath: str = "prompt_comparison.png",
    max_epochs: Optional[int] = None
):
    """
    Plot comparison of original vs optimized prompts over time.
    
    Args:
        log_fpath: Path to the JSON log file
        save_fpath: Path to save the plot image
        max_epochs: Maximum number of epochs to plot (for readability)
    """
    with open(log_fpath, 'r') as f:
        log_data = json.load(f)
    
    if max_epochs:
        log_data = log_data[:max_epochs]
    
    epochs = [entry['epoch'] for entry in log_data]
    
    # Extract prompt texts
    orig_prompts = [entry['orig_prompt']['text'] for entry in log_data]
    optim_prompts = [entry['optim_prompt']['text'] for entry in log_data]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot original prompt
    ax1.plot(epochs, [len(p) for p in orig_prompts], 'b-', linewidth=2)
    ax1.set_ylabel('Original Prompt Length')
    ax1.set_title('Original Prompt Evolution')
    ax1.grid(True, alpha=0.3)
    
    # Plot optimized prompt
    ax2.plot(epochs, [len(p) for p in optim_prompts], 'r-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Optimized Prompt Length')
    ax2.set_title('Optimized Prompt Evolution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_fpath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Prompt comparison saved to: {save_fpath}")


def plot_kl_convergence(
    log_fpaths: List[str],
    labels: List[str],
    save_fpath: str = "kl_convergence.png"
):
    """
    Plot KL convergence for multiple experiments.
    
    Args:
        log_fpaths: List of paths to JSON log files
        labels: List of labels for each experiment
        save_fpath: Path to save the plot image
    """
    plt.figure(figsize=(10, 6))
    
    for log_fpath, label in zip(log_fpaths, labels):
        with open(log_fpath, 'r') as f:
            log_data = json.load(f)
        
        epochs = [entry['epoch'] for entry in log_data]
        kl_values = [entry.get('cur_kl', 0) for entry in log_data]
        
        plt.plot(epochs, kl_values, label=label, linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.title('KL Divergence Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_fpath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"KL convergence plot saved to: {save_fpath}") 