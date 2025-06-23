import numpy as np
from typing import Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.modules.hifigan.parallel_wavegan import SourceModuleHnNS
import logging
get_logger = logging.getLogger

Flogger = get_logger(__name__)

def _get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

def _init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def _apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)

class ResBlock1(torch.nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: Tuple[int] = (1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=_get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=_get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=_get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(_init_weights)
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=_get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=_get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=_get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(_init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x

class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=_get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=_get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(_init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, 0.1)
            xt = c(xt)
            x = xt + x
        return x

class HifiGanGenerator(nn.Module):
    def __init__(self, *, sampling_rate: int, num_mel_bands: int, up_sampling_rates: [int], up_sampling_kernel_sizes: [int], up_sampling_initial_channels: int, res_block_kernel_sizes: [int], res_block_type: str, res_block_dilation_sizes: [[int]], use_pitch_embedding: bool = False, harmonic_num: int = 0, device: str = "cpu"):
        super().__init__()
        self.device = device
        assert res_block_type in ["1", "2"], "res_block_type should be either 1 or 2"
        self.num_kernels = len(res_block_kernel_sizes)
        self.num_up_samples = len(up_sampling_rates)
        if use_pitch_embedding:
            self.source = SourceModuleHnNS(sampling_rate=sampling_rate, harmonic_num=harmonic_num).to(self.device)
            self.noise_convs = nn.ModuleList()
        else:
            self.source = None
        self.conv_pre = weight_norm(Conv1d(num_mel_bands, up_sampling_initial_channels, kernel_size=7, stride=1, padding=3))
        res_block_cls = ResBlock1 if res_block_type == "1" else ResBlock2
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(up_sampling_rates, up_sampling_kernel_sizes)):
            c_cur = up_sampling_initial_channels // (2 ** (i + 1))
            self.ups.append(weight_norm(ConvTranspose1d(up_sampling_initial_channels // (2 ** i), up_sampling_initial_channels // (2 ** (i + 1)), kernel_size=k, stride=u, padding=(k - u) // 2)))
            if use_pitch_embedding:
                if i + 1 < len(up_sampling_rates):
                    stride_f0 = int(np.prod(up_sampling_rates[i + 1:]))
                    self.noise_convs.append(Conv1d(1, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=stride_f0 // 2))
                else:
                    self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))
        self.res_blocks = nn.ModuleList()
        ch = up_sampling_initial_channels
        for i in range(len(self.ups)):
            ch //= 2
            for j, (k, d) in enumerate(zip(res_block_kernel_sizes, res_block_dilation_sizes)):
                self.res_blocks.append(res_block_cls(ch, k, d))
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(_init_weights)
        self.conv_post.apply(_init_weights)
        self.upp = int(np.prod(up_sampling_rates))
        self.to(self.device)

    def forward(self, x: torch.Tensor, f0: torch.Tensor = None):
        if f0 is not None:
            assert self.source is not None, "Pitch embedding is not enabled."
            har_source = self.source(f0, self.upp).transpose(1, 2)
        else:
            har_source = None
        x = self.conv_pre(x)
        for i in range(self.num_up_samples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            if f0 is not None:
                x_source = self.noise_convs[i](har_source)
                x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.res_blocks[i * self.num_kernels + j](x)
                else:
                    xs += self.res_blocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x).squeeze(1)
        x = torch.tanh(x)
        return x