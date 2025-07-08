"""
Soft prompt optimization attacks.
"""

from typing import List, Dict, Tuple, Optional
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import repeat, rearrange
from tqdm import tqdm
import json

from transformers import PreTrainedModel
from ..data import DocDataset
from ..model_wrappers import SoftPromptEmbeddingLayer, OrigModelEmbs
from ..config import SoftPromptConfig


def optim_soft(
    model: PreTrainedModel,
    dataset: DocDataset,
    config: SoftPromptConfig,
    log_fpath: str,
    emb_save_fpath: str,
) -> Tuple[List[Dict], Tensor]:
    """
    Optimize a soft prompt.
    
    Args:
        model: the model
        dataset: dataset to optim
        config: soft prompt configuration
        log_fpath: file to log to
        emb_save_fpath: file to save embeddings to
        
    Returns:
        list of progress logs and best embeddings
    """
    print(
        f"\n\nTRAINING SOFT:\n------------------------\n"
        f"model: {model.config.name_or_path}\n"
        f"num epochs: {config.n_epochs}\n"
        f"kl every: {config.kl_every}\n"
        f"------------------------\n\n"
    )
    
    model.eval()
    to_ret = []
    pbar = tqdm(range(1, config.n_epochs + 1))
    best_embs = model.get_input_embeddings()(
        dataset.wrapped_prompt[:, dataset.prompt_slice]
    ).detach()
    orig_embs = model.get_input_embeddings()
    new_model_embs = SoftPromptEmbeddingLayer(model.get_input_embeddings(), best_embs)
    model.set_input_embeddings(new_model_embs)
    optimizer = torch.optim.Adam(
        [new_model_embs.trainable_embs],
        lr=config.learning_rate,
        eps=1e-4 if model.dtype != torch.float32 else 1e-8,
    )
    
    with OrigModelEmbs(model, orig_embs, new_model_embs):
        from ..evaluation import compute_dataset_kl
        best_kl, best_std = compute_dataset_kl(
            model,
            dataset,
            batch_size=config.batch_size,
            embs=new_model_embs.trainable_embs.detach().clone(),
        )
    cur_kl = None
    cur_std = None
    
    for i in pbar:
        epoch_loss = 0.0
        
        for batch in DataLoader(dataset, batch_size=config.batch_size, shuffle=True):
            with OrigModelEmbs(model, orig_embs, new_model_embs):
                prefix_embs = model.get_input_embeddings()(
                    batch["optim_seq"][:, : dataset.prompt_slice.start]
                )
                suffix_embs = model.get_input_embeddings()(
                    batch["optim_seq"][:, dataset.prompt_slice.stop :]
                )
            
            embs = repeat(
                new_model_embs.trainable_embs,
                "b k d -> (repeat b) k d",
                repeat=prefix_embs.shape[0],
            )
            full_embs = torch.cat(
                [
                    prefix_embs,
                    embs,
                    suffix_embs,
                ],
                dim=1,
            )
            logits = model(inputs_embeds=full_embs).logits
            pred_slice = slice(dataset.doc_slice.start - 1, dataset.doc_slice.stop - 1)
            target_slice = dataset.doc_slice
            
            loss = F.cross_entropy(
                rearrange(logits[:, pred_slice, :], "b k v -> b v k"),
                batch["optim_seq"][:, target_slice],
                reduction="none",
            ).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
        
        if i % config.kl_every == 0:
            with OrigModelEmbs(model, orig_embs, new_model_embs):
                from ..evaluation import compute_dataset_kl
                cur_kl, cur_std = compute_dataset_kl(
                    model,
                    dataset,
                    batch_size=config.batch_size,
                    embs=new_model_embs.trainable_embs.detach().clone(),
                )
            if cur_kl < best_kl:
                best_embs = new_model_embs.trainable_embs.detach().clone()
                best_kl = cur_kl
                best_std = cur_std
                torch.save(best_embs, emb_save_fpath)
        
        to_ret.append(
            {
                "epoch": i,
                "loss": epoch_loss,
                "best_kl": best_kl,
                "best_std": best_std,
                "cur_kl": cur_kl if cur_kl is not None else best_kl,
                "cur_std": cur_std if cur_std is not None else best_std,
            }
        )
        with open(log_fpath, "w") as f:
            json.dump(to_ret, f, indent=4)
        pbar.set_description(
            f"Epoch: {i}; Loss: {epoch_loss:.4f}; Best KL: {best_kl:.4f}; Cur KL: {cur_kl:.4f}"
        )
    
    return to_ret, best_embs 