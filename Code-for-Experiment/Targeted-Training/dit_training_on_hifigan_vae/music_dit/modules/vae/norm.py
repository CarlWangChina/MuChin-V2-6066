import torch
import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.amp.rnn_compat
from typing import Union, List
from einops import rearrange
import torch.nn as nn

class ConvLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape: Union[int, List[int], torch.Size], **kwargs):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'b d ... -> b ... d')
        x = super().forward(x)
        x = rearrange(x, 'b ... d -> b d ...')
        return x