import torch
import torch.nn as nn
from typing import Optional
from einops import rearrange
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.modules.fastspeech2.predictors.predictor_base import PredictorBase

MAX_DURATION = 375.0

class DurationPredictor(PredictorBase):
    def __init__(self, dim: int, num_layers: int = 2, num_channels: int = -1, kernel_size: int = 3, dropout: float = 0.1, offset: float = 1.0, padding_type: str = "same", device: str = "cpu"):
        super(DurationPredictor, self).__init__(dim=dim, num_layers=num_layers, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout, padding_type=padding_type, device=device)
        self.offset = offset
        self.output_proj = nn.Linear(self.num_channels, 1, bias=True, device=self.device)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> (torch.Tensor, torch.Tensor):
        assert x.dim() == 3, f"Input tensor should be in 3D tensor, but got {x.dim()}D tensor."
        assert x.shape[-1] == self.dim, f"The last dimension of the input tensor should be {self.dim}, but got {x.shape[-1]}."
        x = rearrange(x, 'b t d -> b d t')
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        if mask is not None:
            assert mask.shape == (x.shape[0], x.shape[-1]), f"Shape mismatched between mask ({mask.shape}) and input ({x.shape})."
            x *= mask.float()[:, None, :]
        x = rearrange(x, 'b d t -> b t d')
        x = self.output_proj(x)
        if mask is not None:
            x *= mask[:, :, None].float()
        dur = x.squeeze(-1)
        dur = torch.clamp(torch.exp(dur) - self.offset, min=1.0, max=MAX_DURATION)
        return dur, x