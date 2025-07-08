"""
Utility functions for the evil_twins package.

This module contains various utility functions for prompt optimization,
including gradient computation, token replacement, and KL divergence calculation.
"""

from typing import Union, Tuple, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, repeat
from torch.utils.data import DataLoader
import logging

from .datasets import DocDataset

logger = logging.getLogger(__name__)


class OrigModelEmbs:
    """
    Context manager for switching between original and new model embeddings.
    
    This allows temporarily using the original model embeddings while
    keeping the soft prompt embeddings available.
    """
    
    def __init__(
        self, 
        model: PreTrainedModel, 
        orig_embs: nn.Module, 
        new_embs: nn.Module
    ):
        """
        Initialize the context manager.
        
        Args:
            model: The model to switch embeddings for
            orig_embs: Original model embeddings
            new_embs: New model embeddings
        """
        self.model = model
        self.orig_embs = orig_embs
        self.new_embs = new_embs
    
    def __enter__(self):
        """Switch to original embeddings."""
        self.model.set_input_embeddings(self.orig_embs)
    
    def __exit__(self, exception_type, exception_value, exception_traceback):
        """Switch back to new embeddings."""
        self.model.set_input_embeddings(self.new_embs)


def compute_neg_log_prob(
    model: PreTrainedModel, 
    seq: Tensor, 
    pred_slice: slice, 
    target_slice: slice
) -> Tensor:
    """
    Compute negative log probability of a slice in a sequence.
    
    Args:
        model: The model
        seq: Full input sequence (batch_size, n_toks)
        pred_slice: Slice where logits are predicting
        target_slice: Slice with labels for the predicting logits
        
    Returns:
        Negative log probability (n_toks,)
    """
    pred_logits = model(seq).logits[:, pred_slice, :]
    log_probs = -F.cross_entropy(
        rearrange(pred_logits, "b k v -> b v k"), 
        seq[:, target_slice], 
        reduction="none"
    )
    
    return log_probs


def compute_grads(
    model: PreTrainedModel,
    seq: Tensor,
    prompt_slice: slice,
    doc_slice: slice,
    gamma: float = 0.0,
) -> Tensor:
    """
    Compute gradients for each token being optimized.
    
    Args:
        model: The model
        seq: Full sequence (prompt + docs) to compute grad
        prompt_slice: The unwrapped user prompt slice in the sequence
        doc_slice: The doc locations in the sequence
        gamma: Fluency penalty gamma
        
    Returns:
        (n_optim_toks, vocab_size) the grads for each token
    """
    model_embs = model.get_input_embeddings().weight
    
    one_hot_suffix = torch.zeros(
        seq.shape[0],
        prompt_slice.stop - prompt_slice.start,
        model_embs.shape[0],
        device=model.device,
        dtype=model_embs.dtype,
    )
    one_hot_suffix.scatter_(-1, rearrange(seq[:, prompt_slice], "b k -> b k 1"), 1)
    one_hot_suffix.requires_grad = True
    
    suffix_embs = one_hot_suffix @ model_embs
    embs = model.get_input_embeddings()(seq).detach()
    full_embs = torch.cat(
        [
            embs[:, : prompt_slice.start, :],
            suffix_embs,
            embs[:, prompt_slice.stop :, :],
        ],
        dim=1,
    )
    
    logits = model(inputs_embeds=full_embs).logits
    targets = seq[:, doc_slice]
    loss_slice = slice(doc_slice.start - 1, doc_slice.stop - 1)
    
    prompt_pred_slice = slice(prompt_slice.start, prompt_slice.stop - 1)
    prompt_target_slice = slice(prompt_pred_slice.start + 1, prompt_pred_slice.stop + 1)
    
    fluency_penalty = (
        gamma
        * F.cross_entropy(
            rearrange(logits[0, prompt_pred_slice, :], "k v -> 1 v k"),
            rearrange(seq[0, prompt_target_slice], "k -> 1 k"),
            reduction="none",
        )
        .sum(dim=-1)
        .sum(dim=0)
    )
    
    loss = F.cross_entropy(
        rearrange(logits[:, loss_slice, :], "b k v -> b v k"), 
        targets
    )
    loss += fluency_penalty
    loss.backward()
    
    # Mean across docs dim
    return one_hot_suffix.grad.clone().mean(dim=0)


def replace_tok(
    model: PreTrainedModel,
    dataset: DocDataset,
    batch_size: int = 10,
    k: int = 256,
    gamma: float = 0.0,
) -> Tuple[Tensor, float, float]:
    """
    Perform exact loss computations and token replacement.
    
    Args:
        model: The model
        dataset: The dataset
        batch_size: Batch size for doc forward pass
        k: Top-k grads to keep
        gamma: Fluency penalty gamma
        
    Returns:
        The new prompt IDs (1, n_toks) and the best lowest loss and the log prob prompt
    """
    grads = torch.zeros(
        (
            dataset.prompt_slice.stop - dataset.prompt_slice.start,
            model.config.vocab_size,
        ),
        device=model.device,
    )
    
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=True):
        seq = batch["optim_seq"]
        grads += compute_grads(
            model, seq, dataset.prompt_slice, dataset.doc_slice, gamma
        )
    
    grads /= grads.norm(dim=-1, keepdim=True)
    _, top_k_indices = torch.topk(-grads, k=k, dim=-1)
    
    with torch.no_grad():
        # 1 proposal for each position in the optim prompt
        n_proposals = top_k_indices.shape[0]
        grad_indices = rearrange(
            torch.randint(
                0, top_k_indices.shape[-1], (n_proposals,), device=top_k_indices.device
            ),
            "k -> 1 k",
        )
        positions = torch.arange(n_proposals, device=top_k_indices.device)
        new_toks = rearrange(top_k_indices[positions, grad_indices], "b k -> k b")
        positions = rearrange(positions, "k -> k 1")
        proposals = repeat(
            dataset.wrapped_prompt[:, dataset.prompt_slice],
            "b k -> (repeat b) k",
            repeat=n_proposals,
        ).clone()
        proposals = proposals.scatter_(-1, positions, new_toks)
        
        proposal_losses = torch.zeros((n_proposals,), device=model.device)
        
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=True):
            seq = repeat(
                batch["optim_seq"], "b k -> (repeat b) k", repeat=proposals.shape[0]
            ).clone()
            seq[:, dataset.prompt_slice] = repeat(
                proposals,
                "b k -> (repeat b) k",
                repeat=seq.shape[0] // proposals.shape[0],
            )
            
            logits = model(seq).logits
            loss_slice = slice(dataset.doc_slice.start - 1, dataset.doc_slice.stop - 1)
            targets = seq[:, dataset.doc_slice]
            loss = F.cross_entropy(
                rearrange(logits[:, loss_slice, :], "b k v -> b v k"),
                targets,
                reduction="none",
            ).mean(dim=-1)
            
            # Split so that the docs are correctly added
            loss = rearrange(loss, "(k b) -> k b", b=n_proposals)
            loss = loss.sum(dim=0)
            proposal_losses += loss
        
        # Average across ALL docs
        proposal_losses /= len(dataset)
        
        # Factor in fluency penalty
        logits = model(proposals).logits
        nlls = F.cross_entropy(
            rearrange(logits[:, : proposals.shape[-1] - 1, :], "b k v -> b v k"),
            proposals[:, 1:],
            reduction="none",
        ).sum(dim=-1)
        proposals_fluency = gamma * nlls
        proposal_losses += proposals_fluency
        
        best_idx = proposal_losses.argmin(dim=0)
        best_proposal = proposals[best_idx]
        best_loss = proposal_losses.min(dim=0).values
        
        return (
            rearrange(best_proposal, "k -> 1 k"),
            best_loss.item(),
            nlls[best_idx].item(),
        )


@torch.no_grad()
def compute_dataset_kl(
    model: PreTrainedModel,
    dataset: DocDataset,
    batch_size: int,
    embs: Optional[Tensor] = None,
) -> Tuple[float, float]:
    """
    Compute KL divergence between original and optimized prompt using holdout docs.
    
    Args:
        model: The model
        dataset: The dataset
        batch_size: Batch size for doc forward pass
        embs: If using soft prompts, compute from embeddings
        
    Returns:
        Tuple of (KL divergence, standard deviation)
    """
    doc_kls = Tensor([]).to(model.device)
    
    for batch in DataLoader(dataset, batch_size=batch_size):
        orig_pred_slice = slice(
            dataset.orig_doc_slice.start - 1, dataset.orig_doc_slice.stop - 1
        )
        pred_slice = slice(dataset.doc_slice.start - 1, dataset.doc_slice.stop - 1)
        orig_target_slice = dataset.orig_doc_slice
        target_slice = dataset.doc_slice
        
        neg_log_p_orig = compute_neg_log_prob(
            model, batch["orig_seq_dev"], orig_pred_slice, orig_target_slice
        ).sum(dim=-1)
        
        if embs is None:
            neg_log_p = compute_neg_log_prob(
                model, batch["optim_seq_dev"], pred_slice, target_slice
            ).sum(dim=-1)
        else:
            prefix_embs = model.get_input_embeddings()(
                batch["optim_seq_dev"][:, : dataset.prompt_slice.start]
            )
            suffix_embs = model.get_input_embeddings()(
                batch["optim_seq_dev"][:, dataset.prompt_slice.stop :]
            )
            full_embs = torch.cat(
                [
                    prefix_embs,
                    repeat(
                        embs,
                        "b k d -> (repeat b) k d",
                        repeat=prefix_embs.shape[0],
                    ),
                    suffix_embs,
                ],
                dim=1,
            )
            
            logits = model(inputs_embeds=full_embs).logits
            neg_log_p = -F.cross_entropy(
                rearrange(logits[:, pred_slice, :], "b k v -> b v k"),
                batch["optim_seq_dev"][:, target_slice],
                reduction="none",
            ).sum(dim=-1)
        
        cur_kl = neg_log_p_orig - neg_log_p
        doc_kls = torch.cat([doc_kls, cur_kl])
    
    kl = doc_kls.mean().item()
    if kl < 0:
        logger.warning(f"KL < 0: {kl}")
    
    std = doc_kls.std().item() / (doc_kls.shape[0] ** 0.5)
    
    return kl, std 