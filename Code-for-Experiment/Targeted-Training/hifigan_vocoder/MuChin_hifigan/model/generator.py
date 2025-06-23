import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
from typing import List
from omegaconf import DictConfig
from .resnet import ResBlock1, ResBlock2
from .constants import LRELU_SLOPE

class Generator(nn.Module):
    """The generator.    Args:        hparams (DictConfig):               Hyper parameters.        version (str):                      The version of the generator.        upsampling_rates (List[int]):       The upsampling rates of the residual blocks.        upsampling_kernel_sizes (List[int]):The kernel sizes of the upsampling layers.        upsampling_initial_channel (int):   The initial channel of the upsampling layer.        resblock_kernel_sizes (List[int]):  The kernel sizes of the residual blocks.        resblock_dilation_sizes (List[List[int]]):                                            The dilation sizes of the residual blocks.        resblock (str):                     The type of residual block. Should be "1" or "2".  Default: "1".        n_mels (int):                       The number of mel bands. Default: 80.    """
    def __init__(self, *, hparams: DictConfig, version: str, upsampling_rates: List[int], upsampling_kernel_sizes: List[int], upsampling_initial_channel: int, resblock_kernel_sizes: List[int], resblock_dilation_sizes: List[List[int]], resblock: str = "1", n_mels: int = 80):
        super(Generator, self).__init__()
        self.hparams = hparams
        self.version = version
        self.n_mels = n_mels
        assert resblock in ["1", "2"], f"Unrecognized resblock type {resblock}"
        assert len(upsampling_rates) == len(upsampling_kernel_sizes), (f"The length of upsampling_rates ({len(upsampling_rates)}) and " f"upsampling_kernel_sizes ({len(upsampling_kernel_sizes)}) must be the same.")
        assert len(resblock_kernel_sizes) == len(resblock_dilation_sizes), (f"The length of resblock_kernel_sizes ({len(resblock_kernel_sizes)}) and " f"resblock_dilation_sizes ({len(resblock_dilation_sizes)}) must be the same.")
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsampling_rates)
        resblock_cls = ResBlock1 if resblock == "1" else ResBlock2
        self.conv_pre = weight_norm(nn.Conv1d(n_mels, upsampling_initial_channel, kernel_size=7, stride=1, padding=3))
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsampling_rates, upsampling_kernel_sizes)):
            self.ups.append(weight_norm(nn.ConvTranspose1d(upsampling_initial_channel // (2 ** i), upsampling_initial_channel // (2 ** (i + 1)), kernel_size=k, stride=u, padding=(k - u) // 2)))
        self.resblocks = nn.ModuleList()
        ch = 0
        for i in range(len(self.ups)):
            ch = upsampling_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock_cls(num_channels=ch, kernel_size=k, dilation=d))
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        nn.init.kaiming_normal_(self.conv_pre.weight)
        nn.init.zeros_(self.conv_pre.bias)
        for up in self.ups:
            nn.init.kaiming_normal_(up.weight)
            nn.init.zeros_(up.bias)
        nn.init.kaiming_normal_(self.conv_post.weight)
        nn.init.zeros_(self.conv_post.bias)

    def remove_weight_norm_(self):
        for up in self.ups:
            remove_parametrizations(up, "weight")
        for blk in self.resblocks:
            blk.remove_weight_norm_()
        remove_parametrizations(self.conv_pre, "weight")
        remove_parametrizations(self.conv_post, "weight")

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
            x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x