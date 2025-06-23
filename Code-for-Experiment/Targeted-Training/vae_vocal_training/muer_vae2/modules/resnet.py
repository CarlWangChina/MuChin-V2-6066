import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
from typing import List, Optional, Dict, Any
from ama_prof_divi_common.utils import get_logger
from Code_for_Experiment.Targeted_Training.hifigan_vocoder.ama-prof-divi_hifigan.model.utils import get_padding

logger = get_logger(__name__)

class ResBlock(nn.Module):
    def __init__(self,
                 num_channels: int,
                 kernel_size: int,
                 dilations: List[int],
                 activation: str = "LeakyReLU",
                 activation_params: Optional[Dict[str, Any]] = None,
                 norm: str = "BatchNorm1D",
                 norm_params: Optional[Dict[str, Any]] = None,
                 padding_mode="reflect"):
        super(ResBlock, self).__init__()
        activation_params = activation_params or {}

        def get_activation() -> nn.Module:
            return getattr(nn, activation)(**activation_params)

        norm_params = norm_params or {}

        def get_norm() -> nn.Module:
            return getattr(nn, norm)(num_channels, **norm_params)

        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        for dilation in dilations:
            self.convs1.append(nn.Sequential(
                get_activation(),
                weight_norm(nn.Conv1d(
                    in_channels=num_channels,
                    out_channels=num_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=get_padding(kernel_size, dilation),
                    padding_mode=padding_mode
                )),
                get_norm()
            ))
            self.convs2.append(nn.Sequential(
                get_activation(),
                weight_norm(nn.Conv1d(
                    in_channels=num_channels,
                    out_channels=num_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                    padding_mode=padding_mode
                )),
                get_norm()
            ))
        for c1, c2 in zip(self.convs1, self.convs2):
            nn.init.kaiming_normal_(c1[1].weight)
            nn.init.zeros_(c1[1].bias)
            nn.init.kaiming_normal_(c2[1].weight)
            nn.init.zeros_(c2[1].bias)

    def remove_weight_norm_(self):
        for c1, c2 in zip(self.convs1, self.convs2):
            remove_parametrizations(c1[1], "weight")
            remove_parametrizations(c2[1], "weight")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = c1(x)
            xt = c2(xt)
            x = xt + x
        return x

class ResBlockSequence(nn.Module):
    def __init__(self,
                 num_channels: int,
                 kernel_sizes: List[int],
                 dilations: List[List[int]],
                 activation: str = "LeakyReLU",
                 activation_params: Optional[Dict[str, Any]] = None,
                 norm: str = "BatchNorm1D",
                 norm_params: Optional[Dict[str, Any]] = None,
                 padding_mode="reflect"):
        super(ResBlockSequence, self).__init__()
        assert len(kernel_sizes) == len(dilations), "The length of kernel_sizes and dilations should be the same."
        module_list = nn.ModuleList()
        for kernel_size, dilation in zip(kernel_sizes, dilations):
            module_list.append(ResBlock(
                num_channels=num_channels,
                kernel_size=kernel_size,
                dilations=dilation,
                activation=activation,
                activation_params=activation_params,
                norm=norm,
                norm_params=norm_params,
                padding_mode=padding_mode
            ))
        self.layers = nn.Sequential(*module_list)

    def remove_weight_norm_(self):
        for layer in self.layers:
            layer.remove_weight_norm_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)