"""
Model wrapper classes for managing embeddings and soft prompts.
"""

import torch
import torch.nn as nn
from torch import Tensor
from einops import repeat
from transformers import PreTrainedModel


class SoftPromptEmbeddingLayer(nn.Module):
    """
    Replaces the model embedding layer with embedding layer + trainable soft prompts.
    """
    
    def __init__(self, model_embs: nn.Embedding, trainable_embs: Tensor) -> None:
        """
        Initialize the soft prompt embedding layer.
        
        Args:
            model_embs: original model embedding parameters
            trainable_embs: the new trainable soft prompt embeddings `(1, n_toks, d_emb)`
        """
        super().__init__()
        
        self.model_embs = model_embs
        self.trainable_embs = nn.Parameter(trainable_embs)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        New embedding layer w/ added trainable soft prompt.
        
        Args:
            x: token IDs to embed of shape `(batch_size, seq_len)`
            
        Returns:
            Tensor for embedded tokens w/ concat'd trainable soft prompt
        """
        input_embs = self.model_embs(x[:, self.trainable_embs.shape[1] :])
        return torch.cat(
            [
                repeat(
                    self.trainable_embs,
                    "b k d -> (repeat b) k d",
                    repeat=input_embs.shape[0],
                ),
                input_embs,
            ],
            dim=1,
        )


class OrigModelEmbs:
    """
    Context manager to switch to original model embeddings instead of the soft prompt layer.
    """
    
    def __init__(
        self, model: PreTrainedModel, orig_embs: nn.Module, new_embs: nn.Module
    ):
        self.model = model
        self.orig_embs = orig_embs
        self.new_embs = new_embs
    
    def __enter__(self):
        self.model.set_input_embeddings(self.orig_embs)
    
    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.model.set_input_embeddings(self.new_embs) 