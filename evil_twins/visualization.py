"""
Visualization utilities for the evil_twins package.

This module provides functions for plotting training curves and
visualizing optimization results.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import seaborn for better styling
try:
    import seaborn as sns
    sns.set_palette("husl")
    plt.style.use('seaborn-v0_8')
except ImportError:
    # Use default matplotlib style if seaborn is not available
    plt.style.use('default')


class TrainingVisualizer:
    """Visualizer for training curves and optimization results."""
    
    def __init__(self, log_data: List[Dict[str, Any]], output_dir: str = None):
        """
        Initialize the visualizer with training log data.
        
        Args:
            log_data: List of training log entries
            output_dir: Directory to save all outputs (optional)
        """
        self.log_data = log_data
        self.epochs = [entry['epoch'] for entry in log_data]
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_curves(
        self, 
        save_fpath: str = "training_curves.png",
        figsize: tuple = (12, 10),
        dpi: int = 300,
        show_plot: bool = False,
    ) -> None:
        """
        Plot comprehensive training curves.
        
        Args:
            save_fpath: Path to save the plot
            figsize: Figure size (width, height)
            dpi: DPI for the saved image
            show_plot: Whether to display the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Training Curves', fontsize=16, fontweight='bold')
        
        # Plot 1: Loss curves
        self._plot_loss_curves(axes[0, 0])
        
        # Plot 2: KL divergence curves
        self._plot_kl_curves(axes[0, 1])
        
        # Plot 3: NLL Prompt curve
        self._plot_nll_prompt_curve(axes[1, 0])
        
        # Plot 4: Combined metrics
        self._plot_combined_metrics(axes[1, 1])
        
        plt.tight_layout()
        
        # Save plot in output_dir
        save_path = self.output_dir / save_fpath
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Training curves saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def _plot_loss_curves(self, ax: plt.Axes) -> None:
        """Plot loss curves."""
        losses = [entry.get('loss', 0) for entry in self.log_data]
        best_losses = [entry.get('best_loss', 0) for entry in self.log_data]
        
        ax.plot(self.epochs, losses, 'b-', label='Current Loss', linewidth=2, alpha=0.8)
        ax.plot(self.epochs, best_losses, 'r--', label='Best Loss', linewidth=2, alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_kl_curves(self, ax: plt.Axes) -> None:
        """Plot KL divergence curves."""
        best_kls = [entry.get('best_kl', 0) for entry in self.log_data]
        cur_kls = [entry.get('cur_kl', 0) for entry in self.log_data]
        
        ax.plot(self.epochs, best_kls, 'g-', label='Best KL', linewidth=2, alpha=0.8)
        ax.plot(self.epochs, cur_kls, 'orange', label='Current KL', linewidth=2, alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('KL Divergence')
        ax.set_title('KL Divergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_nll_prompt_curve(self, ax: plt.Axes) -> None:
        """Plot NLL prompt curve."""
        nll_prompts = [entry.get('nll_prompt', 0) for entry in self.log_data]
        
        ax.plot(self.epochs, nll_prompts, 'purple', label='NLL Prompt', linewidth=2, alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Negative Log Likelihood')
        ax.set_title('Prompt Negative Log Likelihood')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_combined_metrics(self, ax: plt.Axes) -> None:
        """Plot combined metrics for comparison."""
        losses = [entry.get('loss', 0) for entry in self.log_data]
        cur_kls = [entry.get('cur_kl', 0) for entry in self.log_data]
        
        # Normalize for comparison
        if losses and cur_kls:
            norm_losses = [(l - min(losses)) / (max(losses) - min(losses)) for l in losses]
            norm_kls = [(k - min(cur_kls)) / (max(cur_kls) - min(cur_kls)) for k in cur_kls]
            
            ax.plot(self.epochs, norm_losses, 'b-', label='Normalized Loss', linewidth=2, alpha=0.8)
            ax.plot(self.epochs, norm_kls, 'g-', label='Normalized KL', linewidth=2, alpha=0.8)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Normalized Value')
            ax.set_title('Normalized Metrics Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def plot_prompt_evolution(self, save_fpath: str = "prompt_evolution.png") -> None:
        """
        Plot the evolution of prompts over training.
        
        Args:
            save_fpath: Path to save the plot
        """
        # Extract prompt information
        orig_prompts = [entry.get('orig_prompt', {}).get('text', '') for entry in self.log_data]
        optim_prompts = [entry.get('optim_prompt', {}).get('text', '') for entry in self.log_data]
        
        # Create a simple visualization showing prompt changes
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Original prompt
        ax1.text(0.1, 0.5, f"Original: {orig_prompts[0] if orig_prompts else 'N/A'}", 
                fontsize=12, wrap=True, transform=ax1.transAxes)
        ax1.set_title('Original Prompt')
        ax1.axis('off')
        
        # Optimized prompt (show last one)
        last_optim = optim_prompts[-1] if optim_prompts else 'N/A'
        ax2.text(0.1, 0.5, f"Optimized: {last_optim}", 
                fontsize=12, wrap=True, transform=ax2.transAxes)
        ax2.set_title('Optimized Prompt')
        ax2.axis('off')
        
        plt.tight_layout()
        
        save_path = self.output_dir / save_fpath
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Prompt evolution saved to: {save_path}")
        plt.close()
    
    def create_summary_report(self, save_fpath: str = "training_summary.txt") -> None:
        """
        Create a text summary of the training results.
        
        Args:
            save_fpath: Path to save the summary
        """
        if not self.log_data:
            logger.warning("No log data available for summary")
            return
        
        last_entry = self.log_data[-1]
        
        # Helper function to format numeric values safely
        def format_value(value, default='N/A'):
            if value == 'N/A' or value is None:
                return default
            try:
                return f"{float(value):.4f}"
            except (ValueError, TypeError):
                return str(value)
        
        summary = f"""
Training Summary
================

Model: {last_entry.get('optim_prompt', {}).get('text', 'N/A')}
Total Epochs: {len(self.log_data)}

Final Results:
- Final Loss: {format_value(last_entry.get('loss'))}
- Best Loss: {format_value(last_entry.get('best_loss'))}
- Final KL: {format_value(last_entry.get('cur_kl'))}
- Best KL: {format_value(last_entry.get('best_kl'))}
- Final NLL Prompt: {format_value(last_entry.get('nll_prompt'))}

Original Prompt:
{last_entry.get('orig_prompt', {}).get('text', 'N/A')}

Optimized Prompt:
{last_entry.get('optim_prompt', {}).get('text', 'N/A')}

Training Configuration:
- Batch Size: {last_entry.get('batch_size', 'N/A')}
- Top-k: {last_entry.get('top_k', 'N/A')}
- Gamma: {last_entry.get('gamma', 'N/A')}
"""
        
        save_path = self.output_dir / save_fpath
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            f.write(summary)
        
        logger.info(f"Training summary saved to: {save_path}")


def plot_training_curves(
    log_fpath: str, 
    save_fpath: str = "training_curves.png",
    show_plot: bool = False,
    output_dir: str = None,
) -> None:
    """
    Plot training curves from a log file (legacy function).
    
    Args:
        log_fpath: Path to the JSON log file
        save_fpath: Path to save the plot image
        show_plot: Whether to display the plot
        output_dir: Directory to save outputs
    """
    try:
        with open(log_fpath, 'r') as f:
            log_data = json.load(f)
        
        visualizer = TrainingVisualizer(log_data, output_dir=output_dir)
        visualizer.plot_training_curves(save_fpath, show_plot=show_plot)
        
    except FileNotFoundError:
        logger.error(f"Log file not found: {log_fpath}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in log file: {log_fpath}")
        raise
    except Exception as e:
        logger.error(f"Error plotting training curves: {e}")
        raise


def create_comprehensive_visualization(
    log_fpath: str,
    output_dir: str = "visualization_output",
    show_plots: bool = False,
) -> None:
    """
    Create comprehensive visualization from a log file.
    
    Args:
        log_fpath: Path to the JSON log file
        output_dir: Directory to save all visualizations
        show_plots: Whether to display plots
    """
    try:
        with open(log_fpath, 'r') as f:
            log_data = json.load(f)
        
        visualizer = TrainingVisualizer(log_data, output_dir=output_dir)
        # Create all visualizations
        visualizer.plot_training_curves("training_curves.png", show_plot=show_plots)
        visualizer.plot_prompt_evolution("prompt_evolution.png")
        visualizer.create_summary_report("training_summary.txt")
        logger.info(f"Comprehensive visualization saved to: {visualizer.output_dir}")
        
    except Exception as e:
        logger.error(f"Error creating comprehensive visualization: {e}")
        raise 