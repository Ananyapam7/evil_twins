"""
Dataset and data loading functionality for prompt optimization.
"""

from typing import Union
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from einops import repeat
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from .utils import build_prompt


class DocDataset(Dataset):
    """Dataset for generating and managing documents for prompt optimization."""
    
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
    ) -> None:
        """
        Initialize the document dataset.
        
        Args:
            model: Model to use for generation
            tokenizer: Tokenizer for the model
            orig_prompt: Original prompt (string or tensor)
            optim_prompt: Optimized prompt (string or tensor)
            n_docs: Number of documents to generate
            doc_len: Length of each document
            gen_batch_size: Batch size for generation
            validate_prompt: Whether to validate prompt building
            gen_train_docs: Whether to generate training documents
            gen_dev_docs: Whether to generate development documents
        """
        if isinstance(orig_prompt, str):
            self.orig_wrapped_prompt, self.orig_prompt_slice = build_prompt(
                model.config.name_or_path, orig_prompt, tokenizer, validate_prompt
            )
        else:
            self.orig_wrapped_prompt = orig_prompt
            self.orig_prompt_slice = slice(0, orig_prompt.shape[-1])
        
        self.orig_wrapped_prompt = self.orig_wrapped_prompt.to(model.device)
        self.orig_doc_slice = slice(
            self.orig_wrapped_prompt.shape[-1],
            self.orig_wrapped_prompt.shape[-1] + doc_len,
        )
        
        if isinstance(optim_prompt, str):
            self.wrapped_prompt, self.prompt_slice = build_prompt(
                model.config.name_or_path, optim_prompt, tokenizer, validate_prompt
            )
        else:
            self.wrapped_prompt = optim_prompt
            self.prompt_slice = slice(0, optim_prompt.shape[-1])
        
        self.wrapped_prompt = self.wrapped_prompt.to(model.device)
        self.doc_slice = slice(
            self.wrapped_prompt.shape[-1],
            self.wrapped_prompt.shape[-1] + doc_len,
        )
        
        if not gen_train_docs and not gen_dev_docs:
            print("WARNING: specified NO docs to be generated; continue at your own risk.")
        
        if gen_train_docs:
            self.train_docs = self._gen_docs(model, n_docs, doc_len, gen_batch_size)
        else:
            self.train_docs = None
        
        if gen_dev_docs:
            self.dev_docs = self._gen_docs(model, n_docs, doc_len, gen_batch_size)
        else:
            self.dev_docs = None
    
    def _gen_docs(
        self, model: PreTrainedModel, n_docs: int, doc_len: int, gen_batch_size: int
    ) -> Tensor:
        """
        Generate continuations (with just sampling, no constraints).
        
        Args:
            model: model to gen with
            n_docs: number of continuations to gen
            doc_len: length of each continuation
            gen_batch_size: batch size for gen
            
        Returns:
            doc tokens `(n_docs, doc_len)`
        """
        docs = torch.zeros((n_docs, doc_len), dtype=torch.long, device=model.device)
        
        for i in range(0, n_docs, gen_batch_size):
            cur_bsz = min(gen_batch_size, n_docs - i)
            
            # Without using HF generate method, for consistency
            doc_ids = torch.zeros(
                (cur_bsz, doc_len), dtype=torch.long, device=model.device
            )
            cur_prompt = (
                repeat(self.orig_wrapped_prompt, "b k -> (repeat b) k", repeat=cur_bsz)
                .clone()
                .to(model.device)
            )
            for j in range(doc_len):
                cur_logits = model(cur_prompt).logits
                cur_logits = cur_logits[:, -1, :]
                
                # handle models with logit softcapping (Gemma)
                if (
                    hasattr(model.config, "final_logit_softcapping")
                    and model.config.final_logit_softcapping is not None
                ):
                    cur_logits /= model.config.final_logit_softcapping
                    cur_logits.tanh_()
                    cur_logits *= model.config.final_logit_softcapping
                
                # since we're forcing generation up to doc_len
                cur_logits[..., model.config.eos_token_id] = -float("inf")
                
                cur_probs = F.softmax(cur_logits, dim=-1)
                cur_toks = torch.multinomial(cur_probs, 1)
                doc_ids[:, j] = cur_toks.squeeze(-1)
                cur_prompt = torch.cat([cur_prompt, cur_toks], dim=-1)
            
            docs[i : i + cur_bsz] = doc_ids
        
        return docs
    
    def __len__(self) -> int:
        if self.train_docs is not None:
            return self.train_docs.shape[0]
        if self.dev_docs is not None:
            return self.dev_docs.shape[0]
        
        raise ValueError("No docs generated")
    
    def __getitem__(self, idx: int) -> dict:
        ret = {}
        if self.train_docs is not None:
            ret.update(
                {
                    "optim_seq": torch.cat(
                        [self.wrapped_prompt[0], self.train_docs[idx]], dim=-1
                    ),
                    "orig_seq": torch.cat(
                        [self.orig_wrapped_prompt[0], self.train_docs[idx]], dim=-1
                    ),
                }
            )
        if self.dev_docs is not None:
            ret.update(
                {
                    "optim_seq_dev": torch.cat(
                        [self.wrapped_prompt[0], self.dev_docs[idx]], dim=-1
                    ),
                    "orig_seq_dev": torch.cat(
                        [self.orig_wrapped_prompt[0], self.dev_docs[idx]], dim=-1
                    ),
                }
            )
        
        return ret 