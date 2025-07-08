"""
Model and tokenizer management for the evil_twins package.

This module provides classes and functions for loading and managing
transformer models and tokenizers in a consistent way.
"""

from typing import Union, Tuple, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
import torch
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model and tokenizer loading and configuration."""
    
    def __init__(
        self,
        model_name_or_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device_map: Union[str, dict] = "auto",
        use_flash_attn_2: bool = False,
        eval_mode: bool = True,
        trust_remote_code: bool = True,
    ):
        """
        Initialize the model manager.
        
        Args:
            model_name_or_path: Model path or repo name
            dtype: Torch dtype to load model
            device_map: Device mapping strategy
            use_flash_attn_2: Whether to use flash attention 2
            eval_mode: Whether to set model to eval mode
            trust_remote_code: Whether to trust remote code
        """
        self.model_name_or_path = model_name_or_path
        self.dtype = dtype
        self.device_map = device_map
        self.use_flash_attn_2 = use_flash_attn_2
        self.eval_mode = eval_mode
        self.trust_remote_code = trust_remote_code
        
        self._model: Optional[PreTrainedModel] = None
        self._tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None
    
    def load_model(self) -> PreTrainedModel:
        """
        Load the model.
        
        Returns:
            The loaded model
        """
        if self._model is None:
            logger.info(f"Loading model: {self.model_name_or_path}")
            
            attn_implementation = "flash_attention_2" if self.use_flash_attn_2 else None
            
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                device_map=self.device_map,
                torch_dtype=self.dtype,
                trust_remote_code=self.trust_remote_code,
                attn_implementation=attn_implementation,
            )
            
            if self.eval_mode:
                self._model.eval()
            
            logger.info(f"Model loaded successfully on device: {self._model.device}")
        
        return self._model
    
    def load_tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        """
        Load the tokenizer.
        
        Returns:
            The loaded tokenizer
        """
        if self._tokenizer is None:
            logger.info(f"Loading tokenizer: {self.model_name_or_path}")
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=self.trust_remote_code,
            )
            
            # Set pad token if not present
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            logger.info("Tokenizer loaded successfully")
        
        return self._tokenizer
    
    def load_model_and_tokenizer(
        self,
    ) -> Tuple[PreTrainedModel, Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]:
        """
        Load both model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        model = self.load_model()
        tokenizer = self.load_tokenizer()
        return model, tokenizer
    
    @property
    def model(self) -> PreTrainedModel:
        """Get the loaded model."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._model
    
    @property
    def tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        """Get the loaded tokenizer."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_tokenizer() first.")
        return self._tokenizer
    
    def __enter__(self):
        """Context manager entry."""
        return self.load_model_and_tokenizer()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Clean up if needed
        pass


def load_model_tokenizer(
    model_name_or_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device_map: Union[str, dict] = "auto",
    use_flash_attn_2: bool = False,
    eval_mode: bool = True,
    trust_remote_code: bool = True,
) -> Tuple[PreTrainedModel, Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]:
    """
    Load model and tokenizer (legacy function for backward compatibility).
    
    Args:
        model_name_or_path: Model path or repo name
        dtype: Torch dtype to load model
        device_map: Device mapping strategy
        use_flash_attn_2: Whether to use flash attention 2
        eval_mode: Whether to set model to eval mode
        trust_remote_code: Whether to trust remote code
        
    Returns:
        Tuple of (model, tokenizer)
    """
    manager = ModelManager(
        model_name_or_path=model_name_or_path,
        dtype=dtype,
        device_map=device_map,
        use_flash_attn_2=use_flash_attn_2,
        eval_mode=eval_mode,
        trust_remote_code=trust_remote_code,
    )
    return manager.load_model_and_tokenizer() 