import math
import torch
import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.amp.rnn_compat
import torch.nn.functional as F
from .norm_conv_1d import NormConv1d, NormConvTranspose1d
import torch.nn as nn

def _get_extra_padding_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0) -> int:
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length

def _pad_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0):
    extra_padding = _get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    return F.pad(x, (0, extra_padding))

def _pad1d(x: torch.Tensor, paddings: (int, int), mode: str = 'constant', value: float = 0.):
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
        x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)

def _unpad1d(x: torch.Tensor, paddings: (int, int)):
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]

class StreamableConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int or tuple, stride: int or tuple = 1, padding_mode: str = "reflect", dilation: int or tuple = 1, norm: str = 'none', bias: bool = True):
        super(StreamableConv1d, self).__init__()
        self.conv = NormConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding_mode=padding_mode, dilation=dilation, norm=norm, bias=bias)
        self.padding_mode = padding_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = self.conv.conv.kernel_size[0]
        stride = self.conv.conv.stride[0]
        dilation = self.conv.conv.dilation[0]
        kernel_size = (kernel_size - 1) * dilation + 1
        padding_total = kernel_size - stride
        extra_padding = _get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        padding_right = padding_total // 2
        padding_left = padding_total - padding_right
        x = _pad1d(x, (padding_left, padding_right + extra_padding), mode=self.padding_mode)
        return self.conv(x)

class StreamableConvTranspose1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int or tuple, stride: int or tuple = 1, dilation: int or tuple = 1, norm: str = 'none', bias: bool = True):
        super(StreamableConvTranspose1d, self).__init__()
        self.convtr = NormConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, norm=norm, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = self.convtr.convtr.kernel_size[0]
        stride = self.convtr.convtr.stride[0]
        padding_total = kernel_size - stride
        y = self.convtr(x)
        padding_right = padding_total // 2
        padding_left = padding_total - padding_right
        y = _unpad1d(y, (padding_left, padding_right))
        return y