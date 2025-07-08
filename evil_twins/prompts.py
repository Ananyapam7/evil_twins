"""
Prompt management for the evil_twins package.

This module provides classes and functions for building prompts with
appropriate templates for different model families.
"""

from typing import Union, Tuple, Optional
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import torch
from torch import Tensor
import logging

from .config import ModelConfig

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Handles prompt building with model-specific templates."""
    
    def __init__(self, model_name: str):
        """
        Initialize the prompt builder.
        
        Args:
            model_name: The model name or path
        """
        self.model_name = model_name
        self.template = ModelConfig.get_template(model_name)
        logger.debug(f"Using template '{ModelConfig.get_template_name(model_name)}' for model '{model_name}'")
    
    def build_prompt(
        self,
        prompt: str,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        validate_prompt: bool = True,
    ) -> Tuple[Tensor, slice]:
        """
        Build a prompt with the appropriate template.
        
        Args:
            prompt: The user prompt to wrap
            tokenizer: Tokenizer for the model
            validate_prompt: Whether to validate the prompt slice
            
        Returns:
            Tuple of (prompt_ids, prompt_slice)
        """
        cur_prompt = self.template.prefix
        prompt_start_idx = max(len(tokenizer.encode(cur_prompt)) - 1, 0)
        
        # Account for models that add BOS token
        if prompt_start_idx == 0:
            if tokenizer.encode("test")[0] == tokenizer.bos_token_id:
                # Base model that adds BOS
                prompt_start_idx += 1
        elif (
            tokenizer.encode(cur_prompt)[0] == tokenizer.bos_token_id
            or (
                tokenizer.bos_token_id is None
                and tokenizer.decode(tokenizer.encode(cur_prompt)[0])
                in tokenizer.special_tokens_map.get("additional_special_tokens", [])
            )
        ):
            # Instruct model that also adds some kind of BOS
            prompt_start_idx += 1
        
        cur_prompt += prompt
        prompt_end_idx = len(tokenizer.encode(cur_prompt))
        cur_prompt += self.template.suffix
        
        prompt_ids = tokenizer(cur_prompt, return_tensors="pt").input_ids
        suffix_slice = slice(prompt_start_idx, prompt_end_idx)
        
        if validate_prompt:
            found_prompt = tokenizer.decode(prompt_ids[0, suffix_slice])
            if found_prompt != prompt:
                raise ValueError(
                    f"Prompt building mismatch: {found_prompt} != {prompt}"
                )
        
        return prompt_ids, suffix_slice
    
    def extract_prompt_from_template(self, full_prompt: str) -> str:
        """
        Extract the user prompt from a full wrapped prompt.
        
        Args:
            full_prompt: The full wrapped prompt
            
        Returns:
            The unwrapped user prompt
        """
        pre_loc = full_prompt.find(self.template.prefix) + len(self.template.prefix)
        post_loc = full_prompt.find(self.template.suffix)
        
        if pre_loc == -1 or post_loc == -1:
            raise ValueError(
                f"Could not find template markers in prompt: {full_prompt}"
            )
        
        return full_prompt[pre_loc:post_loc]


def build_prompt(
    model_name: str,
    prompt: str,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    validate_prompt: bool = True,
) -> Tuple[Tensor, slice]:
    """
    Build a prompt with the appropriate template (legacy function).
    
    Args:
        model_name: Model name or path
        prompt: The user prompt to wrap
        tokenizer: Tokenizer for the model
        validate_prompt: Whether to validate the prompt slice
        
    Returns:
        Tuple of (prompt_ids, prompt_slice)
    """
    builder = PromptBuilder(model_name)
    return builder.build_prompt(prompt, tokenizer, validate_prompt)


def extract_prompt_from_template(prompt: str, model_name: str) -> str:
    """
    Extract the user prompt from a full wrapped prompt (legacy function).
    
    Args:
        prompt: Full wrapped prompt
        model_name: Name of model to get the template
        
    Returns:
        Unwrapped user prompt
    """
    builder = PromptBuilder(model_name)
    return builder.extract_prompt_from_template(prompt) 