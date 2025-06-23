import torch
import torch.nn as nn
from typing import Optional

def _get_embedding(num_embeddings: int, embedding_dim: int, scaling_factor: float = 10000.0) -> torch.Tensor:
    weight = torch.zeros(1, num_embeddings, embedding_dim)
    x = (torch.arange(num_embeddings, dtype=torch.float32).reshape(-1, 1) / torch.pow(scaling_factor, torch.arange(0, embedding_dim, 2, dtype=torch.float32) / embedding_dim))
    weight[0, :, 0::2] = torch.sin(x)
    weight[0, :, 1::2] = torch.cos(x)
    return weight

class SinusoidalPosEmbedding(nn.Module):
    def __init__(self, *, dim: int, max_position: int, scaling_factor: float = 10000.0, device: Optional[torch.device] = None):
        super(SinusoidalPosEmbedding, self).__init__()
        self.register_buffer("weight", _get_embedding(num_embeddings=max_position, embedding_dim=dim, scaling_factor=scaling_factor))
        if device is not None:
            self.to(device)

    @torch.no_grad()
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        if positions.dim() == 1:
            return self.weight[:, positions].detach()
        elif positions.dim() == 2:
            return self.weight[:, positions].squeeze(0).detach()