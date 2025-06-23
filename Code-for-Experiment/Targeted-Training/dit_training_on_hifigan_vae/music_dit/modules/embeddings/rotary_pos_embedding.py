import torch
import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.amp.rnn_compat
from typing import Optional

import torch.nn as nn

def _precompute_theta_pos_frequencies(dim: int, max_position: int, scaling_factor: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (scaling_factor ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_position)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cis

def _complex_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.size() == y.size(), f"Tensor x shape {x.size()} must be the same as tensor y shape {y.size()}."
    assert x.size(-1) == 2, f"Tensor x shape {x.size()} must have the last dimension of 2."
    x_real = x[..., 0]
    x_imag = x[..., 1]
    y_real = y[..., 0]
    y_imag = y[..., 1]
    product_real = x_real * y_real - x_imag * y_imag
    product_imag = x_real * y_imag + x_imag * y_real
    product = torch.stack((product_real, product_imag), dim=-1)
    return product

class RotaryPosEmbedding(nn.Module):
    def __init__(self, dim: int, max_position: int, scaling_factor: float = 10000.0, device: Optional[torch.device] = None):
        super(RotaryPosEmbedding, self).__init__()
        self.register_buffer("freqs_cis", _precompute_theta_pos_frequencies(dim=dim, max_position=max_position, scaling_factor=scaling_factor).detach())
        if device is not None:
            self.to(device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, positions: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)
        assert positions.shape == x.shape[:-1], f"Position shape {positions.shape} must be the same as the input shape {x.shape[:-1]}."
        x = x.reshape(*x.shape[:-1], -1, 2)
        freqs_cis = self.freqs_cis[positions.long()]
        x = _complex_product(x, freqs_cis).flatten(-2)
        if mask is not None:
            if mask.dim() == 2:
                x = x * mask.unsqueeze(-1).expand_as(x)
            else:
                assert mask.dim() == 3
                x = x * mask
        return x