import math
import torch
import torch.nn as nn
from typing import Optional
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger

_logger = get_logger(__name__)

def _get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None) -> torch.Tensor:
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
    if embedding_dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    if padding_idx is not None:
        emb[padding_idx, :] = 0
    return emb

def _make_positions(tensor: torch.Tensor, padding_idx: int) -> torch.Tensor:
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, init_size: int = 1024, padding_idx: Optional[int] = None, device: str or torch.device = 'cpu'):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.dim = dim
        self.padding_idx: int = padding_idx if (padding_idx is not None) else 0
        weights = _get_embedding(num_embeddings=init_size, embedding_dim=dim, padding_idx=self.padding_idx).to(device)
        self.register_buffer("weights", weights)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, time_step: torch.Tensor = None, positions: Optional[torch.Tensor] = None, incremental_state: bool = False) -> torch.Tensor:
        assert x.dim() >= 2, f"The input tensor must have at least 2 dimensions.  Got {x.dim()}."
        batch_size, seq_len = x.shape[:2]

        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.shape[0]:
            weights = _get_embedding(num_embeddings=max_pos, embedding_dim=self.dim, padding_idx=self.padding_idx).to(self.device)
            self.register_buffer("weights", weights)
        if incremental_state:
            pos = time_step.view(-1)[0] + 1 if time_step is not None else seq_len
            return self.weights[self.padding_idx + pos, :].expand(batch_size, 1, -1)
        else:
            pos = _make_positions(x, self.padding_idx) if positions is None else positions
            pos = pos.to(self.device)
            return self.weights.index_select(0, pos.view(-1)).view(batch_size, seq_len, -1).detach()

    def embed_on_transformer(self, x: torch.Tensor, start_pos: int = 0, pos_bias: int or list or torch.Tensor = 0) -> torch.Tensor:
        if not torch.is_tensor(pos_bias):
            pos_bias = torch.tensor(pos_bias, dtype=torch.long, device=self.device)
        assert pos_bias.dim() == 0 or pos_bias.dim() == 1
        seq_len = x.shape[1]

        max_pos = self.padding_idx + seq_len + start_pos + torch.max(pos_bias)
        if self.weights is None or max_pos > self.weights.shape[0]:
            weights = _get_embedding(num_embeddings=max_pos, embedding_dim=self.dim, padding_idx=self.padding_idx).to(self.device)
            self.register_buffer("weights", weights)
        if pos_bias.dim() > 0:
            weights = torch.zeros((x.shape[0], seq_len, x.shape[-1]), dtype=x.dtype, device=self.device)
            for i in range(x.shape[0]):
                weights[i] = self.weights[start_pos + pos_bias[i]:start_pos + pos_bias[i] + seq_len]
        else:
            weights = self.weights[start_pos + pos_bias:start_pos + pos_bias + seq_len]
        weights = weights.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = x + weights.detach()
        return x

    @property
    def device(self):
        return self.weights.device