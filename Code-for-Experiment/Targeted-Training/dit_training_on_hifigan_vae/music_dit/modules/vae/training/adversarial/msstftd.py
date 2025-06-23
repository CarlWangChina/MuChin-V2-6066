import torch
import torchaudio
import torch.nn as nn
from typing import Tuple, List, Optional, Dict, Any
from einops import rearrange
from .base import (
    MultiDiscriminator,
    MultiDiscriminatorOutputType,
    LogitsType,
    FeatureMapType
)
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.modules.vae.conv import NormConv2d

def get_2d_padding(kernel_size: Tuple[int, int], dilation: Tuple[int, int] = (1, 1)):
    return (((kernel_size[0] - 1) * dilation[0]) // 2, ((kernel_size[1] - 1) * dilation[1]) // 2)

class DiscriminatorSTFT(nn.Module):
    def __init__(self, filters: int, in_channels: int = 1, out_channels: int = 1, n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024, max_filters: int = 1024, filters_scale: int = 1, kernel_size: Tuple[int, int] = (3, 9), dilations: Optional[List] = None, stride: Tuple[int, int] = (1, 2), normalized: bool = True, norm: str = 'weight_norm', activation_type: str = 'LeakyReLU', activation_params: Optional[Dict[str, Any]] = None):
        super().__init__()
        assert len(kernel_size) == 2
        assert len(stride) == 2
        dilations = dilations or [1, 2, 4]
        activation_params = activation_params or {'negative_slope': 0.2}
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.activation = getattr(torch.nn, activation_type)(**activation_params)
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window_fn=torch.hann_window,
            normalized=self.normalized,
            center=False,
            pad_mode="none",
            power=None
        )
        spec_channels = 2 * self.in_channels
        self.convs = nn.ModuleList()
        self.convs.append(
            NormConv2d(spec_channels, self.filters, kernel_size=kernel_size, padding=get_2d_padding(kernel_size))
        )
        in_chs = min(filters_scale * self.filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale ** (i + 1)) * self.filters, max_filters)
            self.convs.append(NormConv2d(in_chs, out_chs, kernel_size=kernel_size, stride=stride, dilation=(dilation, 1), padding=get_2d_padding(kernel_size, (dilation, 1)), norm=norm))
            in_chs = out_chs
        out_chs = min((filters_scale ** (len(dilations) + 1)) * self.filters, max_filters)
        self.convs.append(NormConv2d(in_chs, out_chs, kernel_size=(kernel_size[0], kernel_size[0]), padding=get_2d_padding((kernel_size[0], kernel_size[0])), norm=norm))
        self.conv_post = NormConv2d(out_chs, self.out_channels, kernel_size=(kernel_size[0], kernel_size[0]), padding=get_2d_padding((kernel_size[0], kernel_size[0])), norm=norm)

    def forward(self, x: torch.Tensor) -> Tuple[LogitsType, FeatureMapType]:
        fmap = []
        z = self.spec_transform(x)
        z = torch.cat([z.real, z.imag], dim=1)
        z = rearrange(z, 'b c w t -> b c t w')
        for i, layer in enumerate(self.convs):
            z = layer(z)
            z = self.activation(z)
            fmap.append(z)
        z = self.conv_post(z)
        return z, fmap

class MultiScaleSTFTDiscriminator(MultiDiscriminator):
    def __init__(self, filters: int, in_channels: int = 1, out_channels: int = 1, sep_channels: bool = False, n_ffts: Optional[List[int]] = None, hop_lengths: Optional[List[int]] = None, win_lengths: Optional[List[int]] = None, **kwargs):
        super().__init__()
        n_ffts = n_ffts or [1024, 2048, 512]
        hop_lengths = hop_lengths or [256, 512, 128]
        win_lengths = win_lengths or [1024, 2048, 512]
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.sep_channels = sep_channels
        self.discriminators = nn.ModuleList([
            DiscriminatorSTFT(filters, in_channels=in_channels, out_channels=out_channels, n_fft=n_ffts[i], win_length=win_lengths[i], hop_length=hop_lengths[i], **kwargs)
            for i in range(len(n_ffts))
        ])

    @property
    def num_discriminators(self):
        return len(self.discriminators)

    @staticmethod
    def _separate_channels(x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        return x.view(-1, 1, T)

    def forward(self, x: torch.Tensor) -> MultiDiscriminatorOutputType:
        logits = []
        fmaps = []
        for disc in self.discriminators:
            logit, fmap = disc(x)
            logits.append(logit)
            fmaps.append(fmap)
        return logits, fmaps