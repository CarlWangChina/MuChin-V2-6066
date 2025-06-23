import torch.nn as nn
import torch.nn.functional as F
from .conv import conv_nd
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.modules.diffusion.unet.norm import avg_pool_nd

class UpSampleBlock(nn.Module):
    def __init__(self, channels: int, out_channels: int = -1, use_conv: bool = True, dims: int = 2):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels if out_channels > 0 else channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class DownSampleBlock(nn.Module):
    def __init__(self, channels: int, out_channels: int = -1, use_conv: bool = True, dims: int = 2):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels if out_channels > 0 else channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)