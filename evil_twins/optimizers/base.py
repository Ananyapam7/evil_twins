"""
Base optimizer interface for the evil_twins package.

This module defines the base class for all optimization strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Union, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
import torch
from torch import Tensor
import json
import logging
from pathlib import Path
import os

from ..datasets import DocDataset
from ..config import OptimizationConfig

logger = logging.getLogger(__name__)


class BaseOptimizer(ABC):
    """
    Base class for all optimization strategies.
    
    This class defines the interface that all optimizers must implement.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        dataset: DocDataset,
        config: OptimizationConfig,
        log_fpath: str,
        output_dir: str = None,
    ):
        """
        Initialize the base optimizer.
        
        Args:
            model: The model to optimize
            tokenizer: The tokenizer for the model
            dataset: The dataset for optimization
            config: Optimization configuration
            log_fpath: Path to save logs (relative to output_dir if provided)
            output_dir: Directory to store all outputs (logs, plots, etc.)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_fpath = self.output_dir / log_fpath
        
        # Training state
        self.best_loss = float("inf")
        self.best_kl = float("inf")
        self.best_std = float("inf")
        self.training_log: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized {self.__class__.__name__} optimizer (output_dir={self.output_dir})")
    
    @abstractmethod
    def optimize(self, n_epochs: int, kl_every: int = 1) -> Tuple[List[Dict[str, Any]], Any]:
        """
        Run the optimization.
        
        Args:
            n_epochs: Number of optimization epochs
            kl_every: How often to compute KL divergence
            
        Returns:
            Tuple of (training_log, best_result)
        """
        pass
    
    @abstractmethod
    def _step(self) -> Dict[str, Any]:
        """
        Perform a single optimization step.
        
        Returns:
            Dictionary containing step results
        """
        pass
    
    def _compute_kl(self) -> Tuple[float, float]:
        """
        Compute KL divergence between original and optimized prompts.
        
        Returns:
            Tuple of (KL divergence, standard deviation)
        """
        from ..utils import compute_dataset_kl
        
        return compute_dataset_kl(
            self.model,
            self.dataset,
            self.config.batch_size,
        )
    
    def _log_step(self, step_info: Dict[str, Any]) -> None:
        """
        Log a training step.
        
        Args:
            step_info: Information about the current step
        """
        # Add common information
        step_info.update({
            "orig_prompt": self.dataset.get_prompt_info()["orig_prompt"],
            "optim_prompt": self.dataset.get_prompt_info()["optim_prompt"],
        })
        
        # Convert tensors to lists for JSON serialization
        serializable_step_info = self._make_serializable(step_info)
        
        self.training_log.append(serializable_step_info)
        
        # Save to file
        with open(self.log_fpath, "w") as f:
            json.dump(self.training_log, f, indent=4, ensure_ascii=False)
        
        logger.debug(f"Logged step: {serializable_step_info}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert objects to JSON-serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable object
        """
        import torch
        import numpy as np
        
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # For other types, try to convert to string
            return str(obj)
    
    def _update_best(self, step_info: Dict[str, Any]) -> None:
        """
        Update best results if current results are better.
        
        Args:
            step_info: Information about the current step
        """
        if "loss" in step_info and step_info["loss"] < self.best_loss:
            self.best_loss = step_info["loss"]
        
        if "kl" in step_info and step_info["kl"] < self.best_kl:
            self.best_kl = step_info["kl"]
            if "std" in step_info:
                self.best_std = step_info["std"]
    
    def get_best_results(self) -> Dict[str, Any]:
        """
        Get the best results achieved during optimization.
        
        Returns:
            Dictionary containing best results
        """
        return {
            "best_loss": self.best_loss,
            "best_kl": self.best_kl,
            "best_std": self.best_std,
        }
    
    def save_checkpoint(self, checkpoint_path: str) -> None:
        """
        Save a checkpoint of the optimizer state.
        
        Args:
            checkpoint_path: Path to save the checkpoint (relative to output_dir)
        """
        checkpoint = {
            "optimizer_type": self.__class__.__name__,
            "config": self.config.to_dict(),
            "best_results": self.get_best_results(),
            "training_log": self.training_log,
        }
        
        checkpoint_path = self.output_dir / checkpoint_path
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=4)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a checkpoint of the optimizer state.
        
        Args:
            checkpoint_path: Path to load the checkpoint from (relative to output_dir)
        """
        checkpoint_path = self.output_dir / checkpoint_path
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        
        # Validate checkpoint type
        if checkpoint["optimizer_type"] != self.__class__.__name__:
            raise ValueError(
                f"Checkpoint type mismatch: expected {self.__class__.__name__}, "
                f"got {checkpoint['optimizer_type']}"
            )
        
        # Load state
        self.config = OptimizationConfig.from_dict(checkpoint["config"])
        self.best_loss = checkpoint["best_results"]["best_loss"]
        self.best_kl = checkpoint["best_results"]["best_kl"]
        self.best_std = checkpoint["best_results"]["best_std"]
        self.training_log = checkpoint["training_log"]
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def __str__(self) -> str:
        """String representation of the optimizer."""
        return f"{self.__class__.__name__}(config={self.config})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the optimizer."""
        return (
            f"{self.__class__.__name__}("
            f"model={self.model.config.name_or_path}, "
            f"config={self.config}, "
            f"log_fpath={self.log_fpath})"
        ) 