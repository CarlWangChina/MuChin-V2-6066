import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from einops import rearrange
from ama_prof_divi.utils.logging import get_logger
from .model_args import WaveNetModelArgs
from . import WaveNetModel
logger = get_logger(__name__)

class ResidualBlock(nn.Module):
    def __init__(self, model_channels: int, context_channels: int, dilation: int, dims: int = 1):
        super(ResidualBlock, self).__init__()
        self.dilated_conv = conv_nd(dims, model_channels, 2 * model_channels, 3, padding=dilation, dilation=dilation)
        self.time_steps_proj = nn.Linear(model_channels, model_channels)
        self.context_proj = conv_nd(dims, context_channels, 2 * model_channels, 1)
        self.out_proj = conv_nd(dims, model_channels, 2 * model_channels, 1)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, time_steps: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        time_steps = self.time_steps_proj(time_steps)
        time_steps = rearrange(time_steps, "b t d -> b d t")
        y = x + time_steps
        y = self.dilated_conv(y)
        if context is not None:
            context = self.context_proj(context)
            y += context
        gate, _filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(_filter)
        y = self.out_proj(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip

class WaveNetModel(nn.Module):
    def __init__(self, args: WaveNetModelArgs, device: str = "cpu"):
        super(WaveNetModel, self).__init__()
        self.args = args
        self.in_proj = conv_nd(self.args.dims, self.args.in_channels, self.args.model_channels, kernel_size=1, device=device)
        self.pos_embedding = SinusoidalPositionalEmbedding(self.args.model_channels, device=device)
        self.mlp = nn.Sequential(
            nn.Linear(self.args.model_channels, self.args.model_channels * 4),
            nn.Mish(),
            nn.Linear(self.args.model_channels * 4, self.args.model_channels)
        ).to(device)
        self.residual_layers = nn.ModuleList([
            ResidualBlock(model_channels=self.args.model_channels, context_channels=self.args.context_channels, dilation=2 ** (i % self.args.dilation_cycle), dims=self.args.dims)
            for i in range(self.args.num_layers)
        ]).to(device)
        self.skip_projection = conv_nd(self.args.dims, self.args.model_channels, self.args.model_channels, kernel_size=1, device=device)
        self.out_projection = conv_nd(self.args.dims, self.args.model_channels, self.args.out_channels, kernel_size=1, device=device)
        nn.init.zeros_(self.out_projection.weight)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.in_proj(x)
        x = F.relu(x)
        if time_steps.dim() == 1:
            time_steps = time_steps.unsqueeze(-1)
        time_steps = self.pos_embedding(time_steps)
        time_steps = self.mlp(time_steps)
        skips = []
        for i, layer in enumerate(self.residual_layers):
            x, skip = layer(x, context=context, time_steps=time_steps)
            skips.append(skip)
        x = torch.sum(torch.stack(skips), dim=0) / math.sqrt(len(skips))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.out_projection(x)
        return x