"""
Soft prompt optimizer for continuous prompt optimization.

This module implements soft prompt optimization by directly optimizing
continuous embeddings rather than discrete tokens.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from einops import rearrange, repeat
from tqdm import tqdm
import logging

from .base import BaseOptimizer
from ..datasets import DocDataset
from ..config import OptimizationConfig
from ..utils import compute_dataset_kl, OrigModelEmbs

logger = logging.getLogger(__name__)


class SoftPromptEmbeddingLayer(nn.Module):
    """
    Replaces the model embedding layer with embedding layer + trainable soft prompts.
    
    This layer allows for continuous optimization of prompt embeddings
    while keeping the original model embeddings fixed.
    """
    
    def __init__(self, model_embs: nn.Embedding, trainable_embs: Tensor) -> None:
        """
        Initialize the soft prompt embedding layer.
        
        Args:
            model_embs: Original model embedding parameters
            trainable_embs: The new trainable soft prompt embeddings (1, n_toks, d_emb)
        """
        super().__init__()
        
        self.model_embs = model_embs
        self.trainable_embs = nn.Parameter(trainable_embs)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with concatenated trainable soft prompt.
        
        Args:
            x: Token IDs to embed of shape (batch_size, seq_len)
            
        Returns:
            Tensor for embedded tokens with concatenated trainable soft prompt
        """
        input_embs = self.model_embs(x[:, self.trainable_embs.shape[1] :])
        return torch.cat(
            [
                repeat(
                    self.trainable_embs,
                    "b k d -> (repeat b) k d",
                    repeat=input_embs.shape[0],
                ),
                input_embs,
            ],
            dim=1,
        )


class SoftPromptOptimizer(BaseOptimizer):
    """
    Soft prompt optimizer for continuous prompt optimization.
    
    This optimizer directly optimizes continuous embeddings rather than
    discrete tokens, allowing for gradient-based optimization.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        dataset: DocDataset,
        config: OptimizationConfig,
        log_fpath: str,
        emb_save_fpath: Optional[str] = None,
        output_dir: str = None,
    ):
        """
        Initialize the soft prompt optimizer.
        
        Args:
            model: The model to optimize
            tokenizer: The tokenizer for the model
            dataset: The dataset for optimization
            config: Optimization configuration
            log_fpath: Path to save logs
            emb_save_fpath: Path to save embeddings (optional)
            output_dir: Directory to save outputs
        """
        super().__init__(model, tokenizer, dataset, config, log_fpath, output_dir)
        self.emb_save_fpath = self.output_dir / emb_save_fpath if emb_save_fpath else None
        
        # Initialize soft prompt embeddings
        self._setup_soft_prompts()
        
        logger.info(f"Initialized soft prompt optimizer")
    
    def _setup_soft_prompts(self):
        """Set up the soft prompt embedding layer and optimizer."""
        # Get initial embeddings from the current prompt
        initial_embs = self.model.get_input_embeddings()(
            self.dataset.wrapped_prompt[:, self.dataset.prompt_slice]
        ).detach()
        
        # Store original embeddings
        self.orig_embs = self.model.get_input_embeddings()
        
        # Create new embedding layer with soft prompts
        self.new_model_embs = SoftPromptEmbeddingLayer(
            self.model.get_input_embeddings(), 
            initial_embs
        )
        
        # Set the new embedding layer
        self.model.set_input_embeddings(self.new_model_embs)
        
        # Create optimizer
        self.optimizer = Adam(
            [self.new_model_embs.trainable_embs],
            lr=self.config.learning_rate,
            eps=1e-4 if self.model.dtype != torch.float32 else 1e-8,
        )
        
        # Store best embeddings
        self.best_embs = initial_embs.clone()
    
    def optimize(
        self, 
        n_epochs: int, 
        kl_every: int = 1,
    ) -> Tuple[List[Dict[str, Any]], Tensor]:
        """
        Run soft prompt optimization.
        
        Args:
            n_epochs: Number of optimization epochs
            kl_every: How often to compute KL divergence
            
        Returns:
            Tuple of (training_log, best_embeddings)
        """
        logger.info(
            f"Starting soft prompt optimization:\n"
            f"  Model: {self.model.config.name_or_path}\n"
            f"  Epochs: {n_epochs}\n"
            f"  KL every: {kl_every}\n"
            f"  Learning rate: {self.config.learning_rate}"
        )
        
        self.model.eval()
        pbar = tqdm(range(1, n_epochs + 1), desc="Soft Prompt Optimization")
        
        # Compute initial KL
        with OrigModelEmbs(self.model, self.orig_embs, self.new_model_embs):
            self.best_kl, self.best_std = compute_dataset_kl(
                self.model,
                self.dataset,
                batch_size=self.config.batch_size,
                embs=self.new_model_embs.trainable_embs.detach().clone(),
            )
        
        cur_kl = None
        cur_std = None
        
        for epoch in pbar:
            # Perform optimization step
            step_info = self._step()
            
            # Compute KL if needed
            if epoch % kl_every == 0:
                with OrigModelEmbs(self.model, self.orig_embs, self.new_model_embs):
                    cur_kl, cur_std = compute_dataset_kl(
                        self.model,
                        self.dataset,
                        batch_size=self.config.batch_size,
                        embs=self.new_model_embs.trainable_embs.detach().clone(),
                    )
                
                step_info.update({"kl": cur_kl, "std": cur_std})
                
                # Update best embeddings if KL improved
                if cur_kl < self.best_kl:
                    self.best_embs = self.new_model_embs.trainable_embs.detach().clone()
                    self.best_kl = cur_kl
                    self.best_std = cur_std
                    
                    # Save embeddings if path provided
                    if self.emb_save_fpath:
                        torch.save(self.best_embs, self.emb_save_fpath)
            
            # Add epoch information
            step_info.update({
                "epoch": epoch,
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
                f"Cur KL: {cur_kl:.4f}" if cur_kl is not None else f"Cur KL: {self.best_kl:.4f}"
            )
        
        return self.training_log, self.best_embs
    
    def _step(self) -> Dict[str, Any]:
        """
        Perform a single soft prompt optimization step.
        
        Returns:
            Dictionary containing step results
        """
        epoch_loss = 0.0
        
        for batch in self.dataset.get_dataloader(self.config.batch_size, shuffle=True):
            with OrigModelEmbs(self.model, self.orig_embs, self.new_model_embs):
                prefix_embs = self.model.get_input_embeddings()(
                    batch["optim_seq"][:, : self.dataset.prompt_slice.start]
                )
                suffix_embs = self.model.get_input_embeddings()(
                    batch["optim_seq"][:, self.dataset.prompt_slice.stop :]
                )
            
            embs = repeat(
                self.new_model_embs.trainable_embs,
                "b k d -> (repeat b) k d",
                repeat=prefix_embs.shape[0],
            )
            full_embs = torch.cat([prefix_embs, embs, suffix_embs], dim=1)
            
            logits = self.model(inputs_embeds=full_embs).logits
            pred_slice = slice(self.dataset.doc_slice.start - 1, self.dataset.doc_slice.stop - 1)
            target_slice = self.dataset.doc_slice
            
            loss = F.cross_entropy(
                rearrange(logits[:, pred_slice, :], "b k v -> b v k"),
                batch["optim_seq"][:, target_slice],
                reduction="none",
            ).sum()
            
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            epoch_loss += loss.item()
        
        return {"loss": epoch_loss}
    
    def get_optimized_embeddings(self) -> Tensor:
        """
        Get the current optimized embeddings.
        
        Returns:
            The optimized embeddings
        """
        return self.new_model_embs.trainable_embs.detach().clone()
    
    def get_best_embeddings(self) -> Tensor:
        """
        Get the best embeddings found during optimization.
        
        Returns:
            The best embeddings
        """
        return self.best_embs.clone()
    
    def decode_embeddings(self, embeddings: Optional[Tensor] = None) -> str:
        """
        Decode embeddings back to text using the model's embedding layer.
        
        Args:
            embeddings: Embeddings to decode (uses current if None)
            
        Returns:
            Decoded text
        """
        if embeddings is None:
            embeddings = self.get_optimized_embeddings()
        
        # Find closest tokens to embeddings
        with torch.no_grad():
            model_embs = self.orig_embs.weight
            distances = torch.cdist(embeddings.squeeze(0), model_embs)
            closest_tokens = torch.argmin(distances, dim=-1)
        
        return self.tokenizer.decode(closest_tokens)


# Legacy function for backward compatibility
def optim_soft(
    model: PreTrainedModel,
    dataset: DocDataset,
    n_epochs: int,
    kl_every: int,
    learning_rate: float,
    log_fpath: str,
    emb_save_fpath: str,
    batch_size: int = 10,
    output_dir: str = None,
) -> Tuple[List[Dict[str, Any]], Tensor]:
    """
    Optimize a soft prompt (legacy function).
    
    Args:
        model: The model
        dataset: Dataset to optimize
        n_epochs: Number of optimization steps
        kl_every: How often to run KL validation
        learning_rate: Learning rate
        log_fpath: File to log to
        emb_save_fpath: File to save embeddings to
        batch_size: Size for doc pass
        output_dir: Directory to save outputs
        
    Returns:
        Tuple of (training_log, best_embeddings)
    """
    config = OptimizationConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    
    optimizer = SoftPromptOptimizer(
        model=model,
        tokenizer=dataset.tokenizer,
        dataset=dataset,
        config=config,
        log_fpath=log_fpath,
        emb_save_fpath=emb_save_fpath,
        output_dir=output_dir,
    )
    
    return optimizer.optimize(n_epochs, kl_every) 