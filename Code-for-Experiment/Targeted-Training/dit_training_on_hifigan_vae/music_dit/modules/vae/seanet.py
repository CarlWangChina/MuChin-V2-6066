import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict, Any
from .conv import SConv1d, SConvTranspose1d
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.modules.vae.lstm import SLSTM

class SEANetResnetBlock(nn.Module):
    def __init__(self,
                 dim: int,
                 kernel_sizes: List[int],
                 dilations: List[int],
                 activation: str = 'ELU',
                 activation_params: Optional[Dict[str, Any]] = None,
                 norm: str = 'weight_norm',
                 norm_params: Optional[Dict[str, Any]] = None,
                 causal: bool = False,
                 pad_mode: str = 'reflect',
                 compress: int = 2,
                 true_skip: bool = True):
        super().__init__()
        assert len(kernel_sizes) == len(dilations), 'Number of kernel sizes should match number of dilations'
        activation_params = activation_params or {"alpha": 1.0}
        norm_params = norm_params or {}
        act = getattr(nn, activation)
        hidden = dim // compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                act(**activation_params),
                SConv1d(in_chs,
                        out_chs,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        norm=norm,
                        norm_kwargs=norm_params,
                        causal=causal,
                        pad_mode=pad_mode),
            ]
        self.block = nn.Sequential(*block)
        self.shortcut: nn.Module
        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = SConv1d(dim,
                                    dim,
                                    kernel_size=1,
                                    norm=norm,
                                    norm_kwargs=norm_params,
                                    causal=causal,
                                    pad_mode=pad_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.block(x)

class SEANetEncoder(nn.Module):
    def __init__(self,
                 channels: int,
                 dimension: int,
                 n_filters: int,
                 n_residual_layers: int,
                 ratios: List[int],
                 activation: str = 'ELU',
                 activation_params: Optional[Dict[str, Any]] = None,
                 norm: str = 'weight_norm',
                 norm_params: Optional[Dict[str, Any]] = None,
                 kernel_size: int = 7,
                 last_kernel_size: int = 7,
                 residual_kernel_size: int = 3,
                 dilation_base: int = 2,
                 causal: bool = False,
                 pad_mode: str = 'reflect',
                 true_skip: bool = False,
                 compress: int = 2,
                 lstm: int = 2):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)
        activation_params = activation_params or {"alpha": 1.0}
        norm_params = norm_params or {}
        act = getattr(nn, activation)
        mult = 1
        model: List[nn.Module] = [
            SConv1d(channels,
                    mult * n_filters,
                    kernel_size,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode)
        ]
        for i, ratio in enumerate(self.ratios):
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(mult * n_filters,
                                      kernel_sizes=[residual_kernel_size, 1],
                                      dilations=[dilation_base ** j, 1],
                                      norm=norm,
                                      norm_params=norm_params,
                                      activation=activation,
                                      activation_params=activation_params,
                                      causal=causal,
                                      pad_mode=pad_mode,
                                      compress=compress,
                                      true_skip=true_skip)]
            model += [
                act(**activation_params),
                SConv1d(mult * n_filters,
                        mult * n_filters * 2,
                        kernel_size=ratio * 2,
                        stride=ratio,
                        norm=norm,
                        norm_kwargs=norm_params,
                        causal=causal,
                        pad_mode=pad_mode),
            ]
            mult *= 2
        if lstm:
            model += [SLSTM(mult * n_filters, num_layers=lstm)]
        model += [
            act(**activation_params),
            SConv1d(mult * n_filters,
                    dimension,
                    last_kernel_size,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode)
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class SEANetDecoder(nn.Module):
    def __init__(self,
                 channels: int,
                 dimension: int,
                 n_filters: int,
                 n_residual_layers: int,
                 ratios: List[int],
                 activation: str = 'ELU',
                 activation_params: Optional[Dict[str, Any]] = None,
                 final_activation: Optional[str] = None,
                 final_activation_params: Optional[dict] = None,
                 norm: str = 'weight_norm',
                 norm_params: Optional[Dict[str, Any]] = None,
                 kernel_size: int = 7,
                 last_kernel_size: int = 7,
                 residual_kernel_size: int = 3,
                 dilation_base: int = 2,
                 causal: bool = False,
                 pad_mode: str = 'reflect',
                 true_skip: bool = False,
                 compress: int = 2,
                 lstm: int = 2,
                 trim_right_ratio: float = 1.0):
        super().__init__()
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)
        activation_params = activation_params or {"alpha": 1.0}
        norm_params = norm_params or {}
        act = getattr(nn, activation)
        mult = int(2 ** len(self.ratios))
        model: List[nn.Module] = [
            SConv1d(dimension, mult * n_filters, kernel_size, norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode)
        ]
        if lstm:
            model += [SLSTM(mult * n_filters, num_layers=lstm)]
        for i, ratio in enumerate(self.ratios):
            model += [
                act(**activation_params),
                SConvTranspose1d(mult * n_filters, mult * n_filters // 2,
                                 kernel_size=ratio * 2, stride=ratio,
                                 norm=norm, norm_kwargs=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
            ]
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(mult * n_filters // 2, kernel_sizes=[residual_kernel_size, 1],
                                      dilations=[dilation_base ** j, 1],
                                      activation=activation, activation_params=activation_params,
                                      norm=norm, norm_params=norm_params, causal=causal,
                                      pad_mode=pad_mode, compress=compress, true_skip=true_skip)]
            mult //= 2
        model += [
            act(**activation_params),
            SConv1d(n_filters, channels, last_kernel_size, norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode)
        ]
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [
                final_act(**final_activation_params)
            ]
        self.model = nn.Sequential(*model)

    def forward(self, z):
        y = self.model(z)
        return y