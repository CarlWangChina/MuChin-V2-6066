import torch
import torch.nn as nn
from einops import rearrange
from ...embeddings import SinusoidalPositionalEmbedding
import SinusoidalPositionalEmbedding
import PredictorBase

class EnergyPredictor(PredictorBase):
    def __init__(self, dim: int, num_layers: int = 2, num_channels: int = -1, kernel_size: int = 3, dropout: float = 0.1, padding_type: str = "same", device: str = "cpu"):
        super(EnergyPredictor, self).__init__(
            dim=dim,
            num_layers=num_layers,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            padding_type=padding_type,
            device=device
        )
        self.output_proj = nn.Linear(self.num_channels, 1, bias=True, device=self.device)
        self.positional_embedding = SinusoidalPositionalEmbedding(dim, init_size=4096, padding_idx=0, device=device)
        self.pos_embed_alpha = nn.Parameter(torch.Tensor([1.0]).to(device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = self.pos_embed_alpha * self.positional_embedding(x[..., 0])
        x = x + positions
        x = rearrange(x, 'b t d -> b d t')
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = rearrange(x, 'b d t -> b t d')
        energy = self.output_proj(x)
        return energy