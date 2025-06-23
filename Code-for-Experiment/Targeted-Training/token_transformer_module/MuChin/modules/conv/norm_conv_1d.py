import torch
import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.amp.rnn
from torch.nn.utils import weight_norm, spectral_norm

def _apply_parametrization_norm(module: nn.Module, norm: str = 'none'):
    if norm == 'weight_norm':
        return weight_norm(module)
    elif norm == 'spectral_norm':
        return spectral_norm(module)
    elif norm == 'none':
        return module
    else:
        raise ValueError(f"Unknown parametrization normalization type: {norm}")

class NormConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int or tuple, stride: int or tuple = 1, padding_mode: str = "reflect", dilation: int or tuple = 1, norm: str = 'none', bias: bool = True):
        super(NormConv1d, self).__init__()
        assert stride == 1 or dilation == 1, \
            f"Unusual situation: stride ({stride}) and dilation ({dilation}) cannot be both greater than 1."
        self.conv = _apply_parametrization_norm(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      dilation=dilation,
                      bias=bias,
                      padding_mode=padding_mode),
            norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class NormConvTranspose1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int or tuple, stride: int or tuple = 1, dilation: int or tuple = 1, norm: str = 'none', bias: bool = True):
        super(NormConvTranspose1d, self).__init__()
        assert stride == 1 or dilation == 1, \
            f"Unusual situation: stride ({stride}) and dilation ({dilation}) cannot be both greater than 1."
        self.convtr = _apply_parametrization_norm(
            nn.ConvTranspose1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               dilation=dilation,
                               bias=bias),
            norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convtr(x)