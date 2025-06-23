import torch
import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.amp.rnn_compat
import math
from typing import Optional
import torch.nn as nn

class TimestepEmbedding(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 frequency_embedding_dim: int = 512,
                 max_period: int = 10000,
                 device: Optional[torch.device] = None):
        super(TimestepEmbedding, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True))
        self.frequency_embedding_dim = frequency_embedding_dim
        self.max_period = max_period
        if device is not None:
            self.to(device)

    @staticmethod
    def timestep_embedding(timesteps: torch.Tensor,
                           dim: int,
                           max_period: int = 10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        )
        freqs = freqs.to(device=timesteps.device)
        timesteps = timesteps.unsqueeze(-1)
        while freqs.dim() < timesteps.dim():
            freqs = freqs.unsqueeze(0)
        args = timesteps.float() * freqs
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self,
                timesteps: torch.Tensor) -> torch.Tensor:
        data_type = next(self.mlp.parameters()).dtype
        t_freq = self.timestep_embedding(timesteps, self.frequency_embedding_dim, self.max_period).to(data_type)
        t_emb = self.mlp(t_freq)
        return t_emb