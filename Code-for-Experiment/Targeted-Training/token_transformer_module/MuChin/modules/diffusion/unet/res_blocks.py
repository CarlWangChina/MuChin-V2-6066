import torch
import torch.nn as nn
from .timestep_block import TimestepBlock
from .norm import normalization, zero_module
from Code-for-Experiment.Targeted-Training.token_transformer_module.ama-prof-divi.modules.diffusion.unet.conv import conv_nd
from .up_down_sampling import UpSampleBlock, DownSampleBlock

class ResBlock(nn.Module):
    def __init__(self,                 channels: int,                 emb_channels: int = 0,                 dropout: float = 0.0,                 out_channels: int = -1,                 use_conv: bool = False,                 use_scale_shift_norm: bool = False,                 dims: int = 1,                 up_sampling: bool = False,                 down_sampling: bool = False,                 ):
        super(ResBlock, self).__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels if out_channels > 0 else channels
        self.up_or_down = up_sampling or down_sampling
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, kernel_size=3, padding=1),
        )
        assert not (up_sampling and down_sampling), \
            "Cannot have both up-sampling and down-sampling in the same block."
        if up_sampling:
            self.h_upd = UpSampleBlock(channels, self.out_channels, use_conv=False, dims=dims)
            self.x_upd = UpSampleBlock(channels, self.out_channels, use_conv=False, dims=dims)
        elif down_sampling:
            self.h_upd = DownSampleBlock(channels, self.out_channels, use_conv=False, dims=dims)
            self.x_upd = DownSampleBlock(channels, self.out_channels, use_conv=False, dims=dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            normalization(self.out_channels),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            )
        )
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self,                x: torch.Tensor,                emb: torch.Tensor = None):
        assert emb is None
        if self.up_or_down:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h

class ResBlockWithTimeEmbedding(TimestepBlock):
    def __init__(self,                 channels: int,                 emb_channels: int,                 dropout: float = 0.0,                 out_channels: int = -1,                 use_conv: bool = False,                 use_scale_shift_norm: bool = False,                 dims: int = 1,                 up_sampling: bool = False,                 down_sampling: bool = False,                 ):
        super(ResBlockWithTimeEmbedding, self).__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels if out_channels > 0 else channels
        self.dropout = dropout
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.updown = up_sampling or down_sampling
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, kernel_size=3, padding=1)
        )
        assert not (up_sampling and down_sampling), \
            "Cannot have both up-sampling and down-sampling in the same block."
        if up_sampling:
            self.h_upd = UpSampleBlock(channels, self.out_channels, use_conv=False, dims=dims)
            self.x_upd = UpSampleBlock(channels, self.out_channels, use_conv=False, dims=dims)
        elif down_sampling:
            self.h_upd = DownSampleBlock(channels, self.out_channels, use_conv=False, dims=dims)
            self.x_upd = DownSampleBlock(channels, self.out_channels, use_conv=False, dims=dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h