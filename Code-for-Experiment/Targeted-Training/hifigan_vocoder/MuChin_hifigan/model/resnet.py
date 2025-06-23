import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
from typing import List
from Code_for_Experiment.Targeted_Training.hifigan_vocoder.ama-prof-divi_hifigan.model.utils import get_padding
from .constants import LRELU_SLOPE

class ResBlock1(nn.Module):
    def __init__(self, num_channels: int, kernel_size: int = 3, dilation: List[int] = (1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, stride=1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, stride=1, dilation=dilation[2], padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, stride=1, dilation=1, padding=get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, stride=1, dilation=1, padding=get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, stride=1, dilation=1, padding=get_padding(kernel_size, 1)))
        ])
        for c1, c2 in zip(self.convs1, self.convs2):
            nn.init.kaiming_normal_(c1.weight)
            nn.init.kaiming_normal_(c2.weight)
            nn.init.zeros_(c1.bias)
            nn.init.zeros_(c2.bias)

    def remove_weight_norm_(self):
        for c1, c2 in zip(self.convs1, self.convs2):
            remove_parametrizations(c1, "weight")
            remove_parametrizations(c2, "weight")

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

class ResBlock2(nn.Module):
    def __init__(self, num_channels: int, kernel_size: int = 3, dilation: List[int] = (1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, stride=1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1])))
        ])
        for c in self.convs:
            nn.init.kaiming_normal_(c.weight)
            nn.init.zeros_(c.bias)

    def remove_weight_norm_(self):
        for c in self.convs:
            remove_parametrizations(c, "weight")

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x