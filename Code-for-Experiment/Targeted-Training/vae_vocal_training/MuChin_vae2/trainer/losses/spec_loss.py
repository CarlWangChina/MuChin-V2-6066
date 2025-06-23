import torch
import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.amp.rnn_compat
import torch.nn as nn
import torch.nn.functional as F
import math
from torchaudio.transforms import MelSpectrogram
from typing import Optional

def get_extra_padding_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0) -> int:
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length

def pad_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0):
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    return F.pad(x, (0, extra_padding))

class MelSpectrogramWrapper(nn.Module):
    def __init__(self,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 win_length: Optional[int] = None,
                 n_mels: int = 80,
                 sampling_rate: int = 44100,
                 f_min: float = 40.0,
                 f_max: Optional[float] = None,
                 log: bool = False,
                 normalized: bool = False,
                 floor_level: float = 1e-5):
        super().__init__()
        self.n_fft = n_fft
        hop_length = int(hop_length)
        self.hop_length = hop_length
        self.mel_transform = MelSpectrogram(n_mels=n_mels,
                                            sample_rate=sampling_rate,
                                            n_fft=n_fft,
                                            hop_length=hop_length,
                                            win_length=win_length,
                                            f_min=f_min,
                                            f_max=f_max,
                                            normalized=normalized,
                                            window_fn=torch.hann_window,
                                            center=False)
        self.floor_level = floor_level
        self.log = log

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, "Input tensor must have shape (batch, channels, samples)."
        p = int((self.n_fft - self.hop_length) // 2)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = F.pad(x, (p, p), "reflect")
        x = pad_for_conv1d(x, self.n_fft, self.hop_length)
        self.mel_transform.to(x.device)
        mel_spec = self.mel_transform(x)
        num_batches, num_channels, num_mel_bins, num_frames = mel_spec.shape
        if self.log:
            mel_spec = torch.log10(self.floor_level + mel_spec)
        return mel_spec.reshape(num_batches, num_channels * num_mel_bins, num_frames)

class MelSpectrogramL1Loss(nn.Module):
    def __init__(self,
                 sampling_rate: int,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 win_length: int = 1024,
                 n_mels: int = 80,
                 f_min: float = 40.0,
                 f_max: Optional[float] = None,
                 log: bool = False,
                 normalized: bool = False,
                 floor_level: float = 1e-5):
        super().__init__()
        self.l1 = torch.nn.L1Loss()
        self.mel_spec = MelSpectrogramWrapper(n_fft=n_fft,
                                              hop_length=hop_length,
                                              win_length=win_length,
                                              n_mels=n_mels,
                                              sampling_rate=sampling_rate,
                                              f_min=f_min,
                                              f_max=f_max,
                                              log=log,
                                              normalized=normalized,
                                              floor_level=floor_level)

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        self.mel_spec.to(x.device)
        s_x = self.mel_spec(x)
        s_y = self.mel_spec(y)
        return self.l1(s_x, s_y)

class MultiScaleMelSpectrogramLoss(nn.Module):
    def __init__(self,
                 sampling_rate: int,
                 range_start: int = 6,
                 range_end: int = 11,
                 n_mels: int = 64,
                 f_min: float = 40.0,
                 f_max: Optional[float] = None,
                 normalized: bool = False,
                 alphas: bool = True,
                 floor_level: float = 1e-5):
        super().__init__()
        l1s = []
        l2s = []
        self.alpha_list = []
        self.total = 0
        self.normalized = normalized
        for i in range(range_start, range_end):
            l1s.append(
                MelSpectrogramWrapper(n_fft=2 ** i,
                                      hop_length=(2 ** i) / 4,
                                      win_length=2 ** i,
                                      n_mels=n_mels,
                                      sampling_rate=sampling_rate,
                                      f_min=f_min,
                                      f_max=f_max,
                                      log=False,
                                      normalized=normalized,
                                      floor_level=floor_level))
            l2s.append(
                MelSpectrogramWrapper(n_fft=2 ** i,
                                      hop_length=(2 ** i) / 4,
                                      win_length=2 ** i,
                                      n_mels=n_mels,
                                      sampling_rate=sampling_rate,
                                      f_min=f_min,
                                      f_max=f_max,
                                      log=True,
                                      normalized=normalized,
                                      floor_level=floor_level))
            if alphas:
                self.alpha_list.append(math.sqrt(2 ** i - 1))
            else:
                self.alpha_list.append(1)
            self.total += self.alpha_list[-1] + 1
        self.l1s = nn.ModuleList(l1s)
        self.l2s = nn.ModuleList(l2s)

    def forward(self, x, y):
        loss = 0.0
        self.l1s.to(x.device)
        self.l2s.to(x.device)
        for i in range(len(self.alpha_list)):
            s_x_1 = self.l1s[i](x)
            s_y_1 = self.l1s[i](y)
            s_x_2 = self.l2s[i](x)
            s_y_2 = self.l2s[i](y)
            loss += F.l1_loss(s_x_1, s_y_1) + self.alpha_list[i] * F.mse_loss(s_x_2, s_y_2)
        if self.normalized:
            loss = loss / self.total
        return loss