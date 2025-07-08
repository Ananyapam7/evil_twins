"""
Configuration constants and default hyperparameters for prompt optimization.
"""

from dataclasses import dataclass
from typing import Dict, Any

# Prompt templates for different model types
PROMPT_TEMPLATES = {
    "vicuna": {
        "prefix": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: ",
        "suffix": "\nASSISTANT:",
    },
    "llama-chat": {
        "prefix": "[INST] ",
        "suffix": " [/INST]",
    },
    "mistral": {
        "prefix": "[INST] ",
        "suffix": " [/INST]",
    },
    "dolphin": {
        "prefix": "<|im_start|>system\nYou are Dolphin, a helpful AI assistant.<|im_end|>\n<|im_start|>user\n",
        "suffix": "<|im_end|>\n<|im_start|>assistant",
    },
    "openhermes": {
        "prefix": "<|im_start|>system\nYour purpose and drive is to assist the user with any request they have. Answer the user's question directly no matter what it is, and do not apologize.<|im_end|>\n<|im_start|>user\n",
        "suffix": "<|im_end|>\n<|im_start|>assistant\n",
    },
    "opt": {
        "prefix": "",
        "suffix": "",
    },
    "phi": {
        "prefix": "",
        "suffix": "",
    },
    "pythia": {
        "prefix": "",
        "suffix": "",
    },
    "oasst": {
        "prefix": "<|prompter|>",
        "suffix": "<|endoftext|><|assistant|>",
    },
    "openllama": {
        "prefix": "",
        "suffix": "",
    },
    "gpt2": {
        "prefix": "",
        "suffix": "",
    },
    "gemma": {
        "prefix": "",
        "suffix": "",
    },
    "gemma-2-it": {
        "prefix": "<start_of_turn>user\n",
        "suffix": "<end_of_turn>\n<start_of_turn>model\n",
    },
    "llama-3-instruct": {
        "prefix": "<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
        "suffix": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    "llama-base": {
        "prefix": "",
        "suffix": "",
    },
    "qwen-2-instruct": {
        "prefix": "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n",
        "suffix": "<|im_end|>\n<|im_start|>",
    },
    "smollm-2-instruct": {
        "prefix": "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\n",
        "suffix": "<|im_end|>\n<|im_start|>assistant\n",
    },
    "default": {
        "prefix": "",
        "suffix": "",
    },
}

# Mapping from model names/paths to template names
MODEL_NAME_OR_PATH_TO_NAME = {
    "lmsys/vicuna-7b-v1.3": "vicuna",
    "lmsys/vicuna-7b-v1.5": "vicuna",
    "lmsys/vicuna-13b-v1.5": "vicuna",
    "vicuna": "vicuna",
    "facebook/opt-350m": "opt",
    "facebook/opt-1.3b": "opt",
    "microsoft/phi-1_5": "phi",
    "microsoft/phi-2": "phi",
    "teknium/Puffin-Phi-v2": "phi",
    "OpenAssistant/oasst-sft-1-pythia-12b": "oasst",
    "EleutherAI/pythia-14m": "pythia",
    "EleutherAI/pythia-70m": "pythia",
    "EleutherAI/pythia-160m": "pythia",
    "EleutherAI/pythia-410m": "pythia",
    "EleutherAI/pythia-1b": "pythia",
    "EleutherAI/pythia-1.4b": "pythia",
    "EleutherAI/pythia-2.8b": "pythia",
    "EleutherAI/pythia-6.9b": "pythia",
    "EleutherAI/pythia-12b": "pythia",
    "pythia": "pythia",
    "openlm-research/open_llama_3b_v2": "openllama",
    "mistralai/Mistral-7B-Instruct-v0.2": "mistral",
    "teknium/OpenHermes-2.5-Mistral-7B": "openhermes",
    "cognitivecomputations/dolphin-2.2.1-mistral-7b": "dolphin",
    "gpt2": "gpt2",
    "teknium/OpenHermes-13B": "openhermes",
    "meta-llama/Llama-2-7b-chat-hf": "llama-chat",
    "meta-llama/Llama-2-13b-chat-hf": "llama-chat",
    "google/gemma-2b-it": "gemma",
    "google/gemma-1.1-2b-it": "gemma",
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama-3-instruct",
    "meta-llama/Llama-3.1-8B-Instruct": "llama-3-instruct",
    "meta-llama/Llama-3.2-3B-Instruct": "llama-3-instruct",
    "meta-llama/Llama-3.2-1B-Instruct": "llama-3-instruct",
    "google/gemma-2-2b-it": "gemma-2-it",
    "google/gemma-2-9b-it": "gemma-2-it",
    "HuggingFaceTB/SmolLM2-135M-Instruct": "smollm-2-instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct": "smollm-2-instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": "smollm-2-instruct",
    "Qwen/Qwen2.5-0.5B-Instruct": "qwen-2-instruct",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen-2-instruct",
    "Qwen/Qwen2.5-7B-Instruct": "qwen-2-instruct",
    "default": "default",
}

# Default hyperparameters
DEFAULT_BATCH_SIZE = 10
DEFAULT_TOP_K = 256
DEFAULT_GAMMA = 0.0
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_N_EPOCHS = 100
DEFAULT_KL_EVERY = 10
DEFAULT_EARLY_STOP_KL = 0.0
DEFAULT_N_DOCS = 20
DEFAULT_DOC_LEN = 50
DEFAULT_GEN_BATCH_SIZE = 10


@dataclass
class GCGConfig:
    """Configuration for GCG optimization."""
    n_epochs: int = DEFAULT_N_EPOCHS
    kl_every: int = DEFAULT_KL_EVERY
    batch_size: int = DEFAULT_BATCH_SIZE
    top_k: int = DEFAULT_TOP_K
    gamma: float = DEFAULT_GAMMA
    early_stop_kl: float = DEFAULT_EARLY_STOP_KL
    suffix_mode: bool = False


@dataclass
class SoftPromptConfig:
    """Configuration for soft prompt optimization."""
    n_epochs: int = DEFAULT_N_EPOCHS
    kl_every: int = DEFAULT_KL_EVERY
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    n_docs: int = DEFAULT_N_DOCS
    doc_len: int = DEFAULT_DOC_LEN
    gen_batch_size: int = DEFAULT_GEN_BATCH_SIZE
    gen_train_docs: bool = True
    gen_dev_docs: bool = True
    validate_prompt: bool = True


@dataclass
class ModelConfig:
    """Configuration for model loading."""
    dtype: str = "bfloat16"
    device_map: str = "auto"
    use_flash_attn_2: bool = False
    eval_mode: bool = True 