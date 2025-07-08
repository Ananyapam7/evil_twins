"""
Evaluation metrics and functions for prompt optimization.
"""

from typing import Optional, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import repeat, rearrange

from transformers import PreTrainedModel
from .data import DocDataset
from .attacks.gcg import compute_neg_log_prob


@torch.no_grad()
def compute_dataset_kl(
    model: PreTrainedModel,
    dataset: DocDataset,
    batch_size: int,
    embs: Optional[Tensor] = None,
) -> Tuple[float, float]:
    """
    Compute KL divergence between original and optimized prompt using the holdout docs.
    
    Args:
        model: the model
        dataset: the dataset
        batch_size: batch size for doc forward pass
        embs: if using soft prompts, compute from embs
        
    Returns:
        kl divergence and standard deviation
    """
    doc_kls = torch.tensor([], device=model.device)
    
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
        print(f"WARNING: KL < 0: {kl}")
    
    std = doc_kls.std().item() / (doc_kls.shape[0] ** 0.5)
    
    return kl, std


def compute_perplexity(
    model: PreTrainedModel,
    dataset: DocDataset,
    batch_size: int,
    embs: Optional[Tensor] = None,
) -> float:
    """
    Compute perplexity of the model on the dataset.
    
    Args:
        model: the model
        dataset: the dataset
        batch_size: batch size for evaluation
        embs: if using soft prompts, compute from embs
        
    Returns:
        perplexity value
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size):
            if embs is None:
                logits = model(batch["optim_seq"]).logits
                targets = batch["optim_seq"][:, dataset.doc_slice]
                pred_slice = slice(dataset.doc_slice.start - 1, dataset.doc_slice.stop - 1)
            else:
                prefix_embs = model.get_input_embeddings()(
                    batch["optim_seq"][:, : dataset.prompt_slice.start]
                )
                suffix_embs = model.get_input_embeddings()(
                    batch["optim_seq"][:, dataset.prompt_slice.stop :]
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
                targets = batch["optim_seq"][:, dataset.doc_slice]
                pred_slice = slice(dataset.doc_slice.start - 1, dataset.doc_slice.stop - 1)
            
            loss = F.cross_entropy(
                logits[:, pred_slice, :].reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += targets.numel()
    
    return torch.exp(torch.tensor(total_loss / total_tokens)).item() 