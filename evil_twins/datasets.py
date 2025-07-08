"""
Dataset management for the evil_twins package.

This module provides dataset classes for prompt optimization,
including document generation and dataset management.
"""

from typing import Union, Dict, Any, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from einops import repeat
import torch.nn.functional as F
import logging

from .prompts import PromptBuilder

logger = logging.getLogger(__name__)


class DocDataset(Dataset):
    """
    Dataset for document generation and prompt optimization.
    
    This dataset generates continuations from prompts and provides
    both training and development splits for optimization.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        orig_prompt: Union[str, Tensor],
        optim_prompt: Union[str, Tensor],
        n_docs: int,
        doc_len: int,
        gen_batch_size: int = 10,
        validate_prompt: bool = True,
        gen_train_docs: bool = True,
        gen_dev_docs: bool = True,
    ):
        """
        Initialize the document dataset.
        
        Args:
            model: The model to use for generation
            tokenizer: The tokenizer for the model
            orig_prompt: Original prompt (string or tensor)
            optim_prompt: Optimization prompt (string or tensor)
            n_docs: Number of documents to generate
            doc_len: Length of each document
            gen_batch_size: Batch size for generation
            validate_prompt: Whether to validate prompt building
            gen_train_docs: Whether to generate training documents
            gen_dev_docs: Whether to generate development documents
        """
        self.model = model
        self.tokenizer = tokenizer
        self.n_docs = n_docs
        self.doc_len = doc_len
        self.gen_batch_size = gen_batch_size
        
        # Build prompts
        self._build_prompts(orig_prompt, optim_prompt, validate_prompt)
        
        # Generate documents
        if not gen_train_docs and not gen_dev_docs:
            logger.warning("No documents specified to be generated; continue at your own risk.")
        
        self.train_docs = None
        self.dev_docs = None
        
        if gen_train_docs:
            self.train_docs = self._gen_docs(n_docs, doc_len, gen_batch_size)
        
        if gen_dev_docs:
            self.dev_docs = self._gen_docs(n_docs, doc_len, gen_batch_size)
    
    def _build_prompts(
        self,
        orig_prompt: Union[str, Tensor],
        optim_prompt: Union[str, Tensor],
        validate_prompt: bool,
    ):
        """Build the original and optimization prompts."""
        model_name = self.model.config.name_or_path
        builder = PromptBuilder(model_name)
        
        # Build original prompt
        if isinstance(orig_prompt, str):
            self.orig_wrapped_prompt, self.orig_prompt_slice = builder.build_prompt(
                orig_prompt, self.tokenizer, validate_prompt
            )
        else:
            self.orig_wrapped_prompt = orig_prompt
            self.orig_prompt_slice = slice(0, orig_prompt.shape[-1])
        
        self.orig_wrapped_prompt = self.orig_wrapped_prompt.to(self.model.device)
        self.orig_doc_slice = slice(
            self.orig_wrapped_prompt.shape[-1],
            self.orig_wrapped_prompt.shape[-1] + self.doc_len,
        )
        
        # Build optimization prompt
        if isinstance(optim_prompt, str):
            self.wrapped_prompt, self.prompt_slice = builder.build_prompt(
                optim_prompt, self.tokenizer, validate_prompt
            )
        else:
            self.wrapped_prompt = optim_prompt
            self.prompt_slice = slice(0, optim_prompt.shape[-1])
        
        self.wrapped_prompt = self.wrapped_prompt.to(self.model.device)
        self.doc_slice = slice(
            self.wrapped_prompt.shape[-1],
            self.wrapped_prompt.shape[-1] + self.doc_len,
        )
    
    def _gen_docs(self, n_docs: int, doc_len: int, gen_batch_size: int) -> Tensor:
        """
        Generate document continuations.
        
        Args:
            n_docs: Number of documents to generate
            doc_len: Length of each document
            gen_batch_size: Batch size for generation
            
        Returns:
            Generated document tokens
        """
        docs = torch.zeros((n_docs, doc_len), dtype=torch.long, device=self.model.device)
        
        for i in range(0, n_docs, gen_batch_size):
            cur_bsz = min(gen_batch_size, n_docs - i)
            
            # Generate docs without using HF generate method for consistency
            doc_ids = torch.zeros(
                (cur_bsz, doc_len), dtype=torch.long, device=self.model.device
            )
            cur_prompt = (
                repeat(self.orig_wrapped_prompt, "b k -> (repeat b) k", repeat=cur_bsz)
                .clone()
                .to(self.model.device)
            )
            
            for j in range(doc_len):
                cur_logits = self.model(cur_prompt).logits
                cur_logits = cur_logits[:, -1, :]
                
                # Handle models with logit softcapping (Gemma)
                if (
                    hasattr(self.model.config, "final_logit_softcapping")
                    and self.model.config.final_logit_softcapping is not None
                ):
                    cur_logits /= self.model.config.final_logit_softcapping
                    cur_logits.tanh_()
                    cur_logits *= self.model.config.final_logit_softcapping
                
                # Prevent EOS token generation
                cur_logits[..., self.model.config.eos_token_id] = -float("inf")
                
                cur_probs = F.softmax(cur_logits, dim=-1)
                cur_toks = torch.multinomial(cur_probs, 1)
                doc_ids[:, j] = cur_toks.squeeze(-1)
                cur_prompt = torch.cat([cur_prompt, cur_toks], dim=-1)
            
            docs[i : i + cur_bsz] = doc_ids
        
        logger.info(f"Generated {n_docs} documents of length {doc_len}")
        return docs
    
    def __len__(self) -> int:
        """Get the number of documents in the dataset."""
        if self.train_docs is not None:
            return self.train_docs.shape[0]
        if self.dev_docs is not None:
            return self.dev_docs.shape[0]
        raise ValueError("No documents generated")
    
    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """
        Get a dataset item.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing the sequences
        """
        ret = {}
        
        if self.train_docs is not None:
            ret.update({
                "optim_seq": torch.cat([self.wrapped_prompt[0], self.train_docs[idx]], dim=-1),
                "orig_seq": torch.cat([self.orig_wrapped_prompt[0], self.train_docs[idx]], dim=-1),
            })
        
        if self.dev_docs is not None:
            ret.update({
                "optim_seq_dev": torch.cat([self.wrapped_prompt[0], self.dev_docs[idx]], dim=-1),
                "orig_seq_dev": torch.cat([self.orig_wrapped_prompt[0], self.dev_docs[idx]], dim=-1),
            })
        
        return ret
    
    def get_dataloader(self, batch_size: int, shuffle: bool = True) -> DataLoader:
        """
        Get a DataLoader for this dataset.
        
        Args:
            batch_size: Batch size for the DataLoader
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader for this dataset
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
    
    def get_prompt_info(self) -> Dict[str, Any]:
        """
        Get information about the prompts in this dataset.
        
        Returns:
            Dictionary containing prompt information
        """
        return {
            "orig_prompt": {
                "text": self.tokenizer.decode(self.orig_wrapped_prompt[0]),
                "ids": self.orig_wrapped_prompt[0].tolist(),
                "prompt_start_slice": self.orig_prompt_slice.start,
                "prompt_end_slice": self.orig_prompt_slice.stop,
                "doc_start_slice": self.orig_doc_slice.start,
                "doc_end_slice": self.orig_doc_slice.stop,
            },
            "optim_prompt": {
                "text": self.tokenizer.decode(self.wrapped_prompt[0]),
                "ids": self.wrapped_prompt[0].tolist(),
                "prompt_start_slice": self.prompt_slice.start,
                "prompt_end_slice": self.prompt_slice.stop,
                "doc_start_slice": self.doc_slice.start,
                "doc_end_slice": self.doc_slice.stop,
            },
        } 