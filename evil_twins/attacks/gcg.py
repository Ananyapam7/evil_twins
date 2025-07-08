"""
Gradient-based optimization attacks for hard prompt optimization.
"""

from typing import List, Dict, Tuple, Union
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange, repeat
from tqdm import tqdm
import json

from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from ..data import DocDataset
from ..config import GCGConfig


def compute_neg_log_prob(
    model: PreTrainedModel, seq: Tensor, pred_slice: slice, target_slice: slice
) -> Tensor:
    """
    Compute negative log prob of a slice in a sequence.
    
    Args:
        model: the model
        seq: full input sequence `(batch_size, n_toks)`
        pred_slice: slice where logits are predicting
        target_slice: slice with labels for the predicting logits
        
    Returns:
        negative log probability `(n_toks,)`
    """
    pred_logits = model(seq).logits[:, pred_slice, :]
    log_probs = -F.cross_entropy(
        rearrange(pred_logits, "b k v -> b v k"), seq[:, target_slice], reduction="none"
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
        model: the model
        seq: full sequence (prompt + docs) to compute grad
        prompt_slice: the unwrapped user prompt slice in the sequence
        doc_slice: the doc locations in the sequence
        gamma: fluency penalty gamma
        
    Returns:
        `(n_optim_toks, vocab_size)` the grads for each token
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
    prompt_target_slice = slice(
        prompt_pred_slice.start + 1, prompt_pred_slice.stop + 1
    )
    fluency_penalty = (
        gamma
        * F.cross_entropy(
            rearrange(logits[0, prompt_pred_slice, :], "k v -> 1 v k"),
            rearrange(seq[0, prompt_target_slice], "k -> 1 k"),
            reduction="none",
        )
        .sum(dim=-1)
        .sum(dim=0)
    )  # sum over all tokens in prompt (only take first [0] since the log prob prompt is same for all docs )
    
    loss = F.cross_entropy(
        rearrange(logits[:, loss_slice, :], "b k v -> b v k"), targets
    )
    loss += fluency_penalty
    loss.backward()
    
    # mean across docs dim
    return one_hot_suffix.grad.clone().mean(dim=0)  # type: ignore


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
        model: the model
        dataset: the dataset
        batch_size: batch size for doc forward pass
        k: top-k grads to keep
        gamma: fluency penalty gamma
        
    Returns:
        the new prompt IDs `(1, n_toks)` and the best lowest loss and the log prob. prompt
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
            # Now we have batch [replaced_tok_1 + doc_1, replaced_tok_2 + doc_2, ...]
            # for each n_proposals * batch_size
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
            # Split so that the docs are correctly added (split by num proposals, then sum across the docs)
            loss = rearrange(loss, "(k b) -> k b", b=n_proposals)
            loss = loss.sum(dim=0)
            proposal_losses += loss
        
        # avg across ALL docs
        proposal_losses /= len(dataset)
        
        # Factor in fluency penalty
        logits = model(proposals).logits
        nlls = F.cross_entropy(
            rearrange(logits[:, : proposals.shape[-1] - 1, :], "b k v -> b v k"),
            proposals[:, 1:],
            reduction="none",
        ).sum(dim=-1)  # sum over all tokens in prompt
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


def optim_gcg(
    model: PreTrainedModel,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    dataset: DocDataset,
    config: GCGConfig,
    log_fpath: str,
) -> Tuple[List[Dict], Tensor]:
    """
    Optimize a hard prompt via GCG.
    
    Args:
        model: the model
        tokenizer: tokenizer
        dataset: the document/prompt dataset
        config: GCG configuration
        log_fpath: file for logging progress
        
    Returns:
        list of progress log/results, and best optimized IDs `(1, n_optim_toks)`
    """
    print(
        f"\n\nTRAINING GCG:\n------------------------\n"
        f"model: {model.config.name_or_path}\n"
        f"num epochs: {config.n_epochs}\n"
        f"kl every: {config.kl_every}\n"
        f"gamma: {config.gamma}\n"
        f"early stopping KL: {config.early_stop_kl}\n"
        f"------------------------\n\n"
    )
    
    if config.suffix_mode:
        assert (
            dataset.train_docs.shape[0] == 1
        ), "Suffix mode should only have 1 doc: the suffix to optimize for"
    
    model.eval()
    pbar = tqdm(range(1, config.n_epochs + 1))
    to_ret = []
    best_loss = float("inf")
    if not config.suffix_mode:
        from ..evaluation import compute_dataset_kl
        best_kl, best_std = compute_dataset_kl(model, dataset, batch_size=10)
    best_ids = dataset.wrapped_prompt[:, dataset.prompt_slice]
    cur_kl = None
    cur_std = None
    
    for i in pbar:
        ids, loss, log_prob_prompt = replace_tok(
            model, dataset=dataset, batch_size=config.batch_size, 
            k=config.top_k, gamma=config.gamma
        )
        dataset.wrapped_prompt[:, dataset.prompt_slice] = ids
        
        if config.suffix_mode and loss < best_loss:
            best_ids = ids
            best_loss = loss
        elif not config.suffix_mode and i % config.kl_every == 0:
            from ..evaluation import compute_dataset_kl
            cur_kl, cur_std = compute_dataset_kl(model, dataset, batch_size=config.batch_size)
            if cur_kl < best_kl:
                best_ids = ids
                best_kl = cur_kl
                best_std = cur_std
        
        best_loss = min(loss, best_loss)
        
        to_ret.append(
            {
                "epoch": i,
                "loss": loss,
                "best_loss": best_loss,
                "best_kl": best_kl,
                "best_std": best_std,
                "cur_kl": cur_kl if cur_kl is not None else best_kl,
                "cur_std": cur_std if cur_std is not None else best_std,
                "orig_prompt": {
                    "text": tokenizer.decode(dataset.orig_wrapped_prompt[0]),
                    "ids": dataset.orig_wrapped_prompt[0].tolist(),
                    "prompt_start_slice": dataset.orig_prompt_slice.start,
                    "prompt_end_slice": dataset.orig_prompt_slice.stop,
                    "doc_start_slice": dataset.orig_doc_slice.start,
                    "doc_end_slice": dataset.orig_doc_slice.stop,
                },
                "optim_prompt": {
                    "text": tokenizer.decode(dataset.wrapped_prompt[0]),
                    "ids": dataset.wrapped_prompt[0].tolist(),
                    "prompt_start_slice": dataset.prompt_slice.start,
                    "prompt_end_slice": dataset.prompt_slice.stop,
                    "doc_start_slice": dataset.doc_slice.start,
                    "doc_end_slice": dataset.doc_slice.stop,
                },
                "nll_prompt": -log_prob_prompt,
            }
        )
        pbar.set_description(
            f"Epoch: {i}; Loss: {loss:.4f}; Best KL: {best_kl:.4f}; "
            f"Cur KL: {(cur_kl if cur_kl is not None else best_kl):.4f}; "
            f"NLL Prompt: {-log_prob_prompt:.4f}"
        )
        with open(log_fpath, "w") as f:
            json.dump(to_ret, f, indent=4, ensure_ascii=False)
        
        if best_kl < config.early_stop_kl:
            print(f"Early KL stopping <{config.early_stop_kl}")
            return to_ret, best_ids
    
    return to_ret, best_ids 