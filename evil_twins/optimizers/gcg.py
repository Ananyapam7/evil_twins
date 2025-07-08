"""
GCG (Gradient-based Controlled Generation) optimizer for hard prompt optimization.

This module implements the GCG algorithm for optimizing discrete token sequences.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
import torch
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange, repeat
from tqdm import tqdm
import logging

from .base import BaseOptimizer
from ..datasets import DocDataset
from ..config import OptimizationConfig
from ..utils import compute_grads, replace_tok

logger = logging.getLogger(__name__)


class GCGOptimizer(BaseOptimizer):
    """
    GCG optimizer for hard prompt optimization.
    
    This optimizer uses gradient-based controlled generation to optimize
    discrete token sequences by iteratively replacing tokens based on
    gradient information.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        dataset: DocDataset,
        config: OptimizationConfig,
        log_fpath: str,
        suffix_mode: bool = False,
        output_dir: str = None,
    ):
        """
        Initialize the GCG optimizer.
        
        Args:
            model: The model to optimize
            tokenizer: The tokenizer for the model
            dataset: The dataset for optimization
            config: Optimization configuration
            log_fpath: Path to save logs
            suffix_mode: Whether to optimize for a single suffix
            output_dir: Directory for saving outputs
        """
        super().__init__(model, tokenizer, dataset, config, log_fpath, output_dir)
        self.suffix_mode = suffix_mode
        
        if suffix_mode:
            if dataset.train_docs.shape[0] != 1:
                raise ValueError("Suffix mode should only have 1 doc: the suffix to optimize for")
        
        logger.info(f"Initialized GCG optimizer with suffix_mode={suffix_mode}")
    
    def optimize(
        self, 
        n_epochs: int, 
        kl_every: int = 1,
        early_stop_kl: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], Tensor]:
        """
        Run GCG optimization.
        
        Args:
            n_epochs: Number of optimization epochs
            kl_every: How often to compute KL divergence
            early_stop_kl: Early stopping KL threshold
            
        Returns:
            Tuple of (training_log, best_prompt_ids)
        """
        logger.info(
            f"Starting GCG optimization:\n"
            f"  Model: {self.model.config.name_or_path}\n"
            f"  Epochs: {n_epochs}\n"
            f"  KL every: {kl_every}\n"
            f"  Gamma: {self.config.gamma}\n"
            f"  Early stop KL: {early_stop_kl}\n"
            f"  Suffix mode: {self.suffix_mode}"
        )
        
        self.model.eval()
        pbar = tqdm(range(1, n_epochs + 1), desc="GCG Optimization")
        
        best_ids = self.dataset.wrapped_prompt[:, self.dataset.prompt_slice].clone()
        cur_kl = None
        cur_std = None
        
        # Compute initial KL if not in suffix mode
        if not self.suffix_mode:
            self.best_kl, self.best_std = self._compute_kl()
        
        for epoch in pbar:
            # Perform optimization step
            step_info = self._step()
            
            # Update dataset with new prompt
            self.dataset.wrapped_prompt[:, self.dataset.prompt_slice] = step_info["ids"]
            
            # Handle best result tracking
            if self.suffix_mode and step_info["loss"] < self.best_loss:
                best_ids = step_info["ids"].clone()
                self.best_loss = step_info["loss"]
            elif not self.suffix_mode and epoch % kl_every == 0:
                cur_kl, cur_std = self._compute_kl()
                step_info.update({"kl": cur_kl, "std": cur_std})
                
                if cur_kl < self.best_kl:
                    best_ids = step_info["ids"].clone()
                    self.best_kl = cur_kl
                    self.best_std = cur_std
            
            # Update best loss
            self.best_loss = min(step_info["loss"], self.best_loss)
            
            # Add epoch information
            step_info.update({
                "epoch": epoch,
                "best_loss": self.best_loss,
                "best_kl": self.best_kl,
                "best_std": self.best_std,
                "cur_kl": cur_kl if cur_kl is not None else self.best_kl,
                "cur_std": cur_std if cur_std is not None else self.best_std,
            })
            
            # Log step
            self._log_step(step_info)
            self._update_best(step_info)
            
            # Update progress bar
            pbar.set_description(
                f"Epoch: {epoch}; Loss: {step_info['loss']:.4f}; "
                f"Best KL: {self.best_kl:.4f}; "
                f"Cur KL: {(cur_kl if cur_kl is not None else self.best_kl):.4f}; "
                f"NLL Prompt: {-step_info['log_prob_prompt']:.4f}"
            )
            
            # Early stopping
            if early_stop_kl is not None and self.best_kl < early_stop_kl:
                logger.info(f"Early stopping triggered: KL < {early_stop_kl}")
                break
        
        return self.training_log, best_ids
    
    def _step(self) -> Dict[str, Any]:
        """
        Perform a single GCG optimization step.
        
        Returns:
            Dictionary containing step results
        """
        ids, loss, log_prob_prompt = replace_tok(
            model=self.model,
            dataset=self.dataset,
            batch_size=self.config.batch_size,
            k=self.config.top_k,
            gamma=self.config.gamma,
        )
        
        return {
            "ids": ids,
            "loss": loss,
            "log_prob_prompt": log_prob_prompt,
        }
    
    def get_optimized_prompt(self) -> str:
        """
        Get the current optimized prompt as text.
        
        Returns:
            The optimized prompt text
        """
        return self.tokenizer.decode(
            self.dataset.wrapped_prompt[0, self.dataset.prompt_slice]
        )
    
    def get_optimized_tokens(self) -> Tensor:
        """
        Get the current optimized prompt as token IDs.
        
        Returns:
            The optimized prompt token IDs
        """
        return self.dataset.wrapped_prompt[0, self.dataset.prompt_slice].clone()


# Legacy function for backward compatibility
def optim_gcg(
    model: PreTrainedModel,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    dataset: DocDataset,
    n_epochs: int,
    kl_every: int,
    log_fpath: str,
    batch_size: int = 10,
    top_k: int = 256,
    gamma: float = 0.0,
    early_stop_kl: float = 0.0,
    suffix_mode: bool = False,
    output_dir: str = None,
) -> Tuple[List[Dict[str, Any]], Tensor]:
    """
    Optimize a hard prompt via GCG (legacy function).
    
    Args:
        model: The model
        tokenizer: Tokenizer
        dataset: The document/prompt dataset
        n_epochs: Number of epochs
        kl_every: How often to compute KL
        log_fpath: File for logging progress
        batch_size: Batch size for docs forward pass
        top_k: Top-k for keeping in gradients
        gamma: Natural prompt (fluency) penalty
        early_stop_kl: If KL goes below this threshold, stop optimization
        suffix_mode: If True, optimize a single document for a suffix
        output_dir: Directory for saving outputs
        
    Returns:
        List of progress log/results, and best optimized IDs
    """
    config = OptimizationConfig(
        batch_size=batch_size,
        top_k=top_k,
        gamma=gamma,
        early_stop_kl=early_stop_kl,
    )
    
    optimizer = GCGOptimizer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        config=config,
        log_fpath=log_fpath,
        suffix_mode=suffix_mode,
        output_dir=output_dir,
    )
    
    return optimizer.optimize(n_epochs, kl_every, early_stop_kl) 