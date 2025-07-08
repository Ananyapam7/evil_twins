"""
Utility functions for prompt optimization.
"""

from typing import Union, Tuple
import torch
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from einops import repeat

from .config import PROMPT_TEMPLATES, MODEL_NAME_OR_PATH_TO_NAME


def extract_prompt_from_template(prompt: str, model_name: str) -> str:
    """
    Extract the unwrapped user prompt from a full wrapped prompt.
    
    Args:
        prompt: full wrapped prompt
        model_name: name of model to get the template
        
    Returns:
        unwrapped user prompt
    """
    prefix = PROMPT_TEMPLATES[MODEL_NAME_OR_PATH_TO_NAME[model_name]]["prefix"]
    suffix = PROMPT_TEMPLATES[MODEL_NAME_OR_PATH_TO_NAME[model_name]]["suffix"]
    pre_loc = prompt.find(prefix) + len(prefix)
    post_loc = prompt.find(suffix)
    
    return prompt[pre_loc:post_loc]


def build_prompt(
    model_name: str,
    prompt: str,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    validate_prompt: bool = True,
) -> Tuple[Tensor, slice]:
    """
    Given the actual user prompt, add in the prefix/suffix for the given instruction tuned model.
    
    Args:
        model_name: Model name or path
        prompt: The actual prompt to wrap around
        tokenizer: Tokenizer for the model
        validate_prompt: Ensure the prompt slice we found is exactly the original prompt
        
    Returns:
        Tuple of the prompt ids `(1, n_toks)` and the slice of the actual prompt (suffix)
    """
    if model_name not in MODEL_NAME_OR_PATH_TO_NAME:
        # first try to match it
        found = False
        for key in MODEL_NAME_OR_PATH_TO_NAME:
            if key.split("/")[-1] in model_name:
                model_name = key
                print(f"Custom path provided, using model name: {model_name}")
                found = True
                break
                
        if not found:
            print(f"Model {model_name} name not found, using default (no template)")
            model_name = "default"
    
    model_name = MODEL_NAME_OR_PATH_TO_NAME[model_name]
    cur_prompt = PROMPT_TEMPLATES[model_name]["prefix"]
    
    prompt_start_idx = max(len(tokenizer.encode(cur_prompt)) - 1, 0)
    
    # account for models that add BOS token (account for models that dont have BOS but do have <|im_start|>)
    if prompt_start_idx == 0:
        if tokenizer.encode("test")[0] == tokenizer.bos_token_id:
            # base model that adds BOS
            prompt_start_idx += 1
    elif tokenizer.encode(cur_prompt)[0] == tokenizer.bos_token_id or (
        tokenizer.bos_token_id is None
        and tokenizer.decode(tokenizer.encode(cur_prompt)[0])
        in tokenizer.special_tokens_map["additional_special_tokens"]
    ):
        # instruct model that also adds some kind of BOS
        prompt_start_idx += 1
    
    cur_prompt += prompt
    prompt_end_idx = len(tokenizer.encode(cur_prompt))
    cur_prompt += PROMPT_TEMPLATES[model_name]["suffix"]
    
    prompt_ids = tokenizer(cur_prompt, return_tensors="pt").input_ids
    suffix_slice = slice(prompt_start_idx, prompt_end_idx)
    
    if validate_prompt:
        found_prompt = tokenizer.decode(prompt_ids[0, suffix_slice])
        
        assert (
            found_prompt == prompt
        ), f"Prompt building mismatch: {found_prompt} != {prompt}"
    
    return prompt_ids, suffix_slice


def load_model_tokenizer(
    model_name_or_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device_map: Union[str, dict] = "auto",
    use_flash_attn_2: bool = False,
    eval_mode: bool = True,
) -> Tuple[PreTrainedModel, Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]:
    """
    Load model and tokenizer.
    
    Args:
        model_name_or_path: model path or repo name
        dtype: torch dtype to load model
        device_map: device to use
        use_flash_attn_2: only compatible with transformers >= 4.36.0
        
    Returns:
        the loaded model and tokenizer
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if use_flash_attn_2 else None,
    )
    if eval_mode:
        model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    return model, tokenizer


def get_dtype_from_string(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.bfloat16) 