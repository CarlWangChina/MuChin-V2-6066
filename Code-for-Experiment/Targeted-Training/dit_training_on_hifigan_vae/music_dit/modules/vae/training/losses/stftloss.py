import torch
import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.amp.rnn_compat
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Sequence

def _stft(x: torch.Tensor,
          fft_size: int,
          hop_length: int,
          win_length: int,
          window: Optional[torch.Tensor] = None,
          normalized: bool = False,
          eps: float = torch.finfo(torch.float32).eps) -> torch.Tensor:
    B, C, T = x.shape
    x_stft = torch.stft(
        x.view(-1, T), fft_size, hop_length, win_length, window,
        normalized=normalized, return_complex=True,
    )
    x_stft = x_stft.view(B, C, *x_stft.shape[1:])
    real = x_stft.real
    imag = x_stft.imag
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=eps)).transpose(2, 1)

class SpectralConvergenceLoss(nn.Module):
    def __init__(self, epsilon: float = torch.finfo(torch.float32).eps):
        super().__init__()
        self.epsilon = epsilon

    def forward(self,
                x_mag: torch.Tensor,
                y_mag: torch.Tensor):
        return torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + self.epsilon)

class LogSTFTMagnitudeLoss(nn.Module):
    def __init__(self,
                 epsilon: float = torch.finfo(torch.float32).eps):
        super().__init__()
        self.epsilon = epsilon

    def forward(self,
                x_mag: torch.Tensor,
                y_mag: torch.Tensor):
        return F.l1_loss(torch.log(self.epsilon + y_mag), torch.log(self.epsilon + x_mag))

class STFTLosses(nn.Module):
    def __init__(self,
                 n_fft: int = 1024,
                 hop_length: int = 120,
                 win_length: int = 600,
                 window: str = "hann_window",
                 normalized: bool = False,
                 epsilon: float = torch.finfo(torch.float32).eps):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.spectral_convergence_loss = SpectralConvergenceLoss(epsilon)
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss(epsilon)

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_mag = _stft(x, self.n_fft, self.hop_length,
                      self.win_length, self.window, self.normalized)
        y_mag = _stft(y, self.n_fft, self.hop_length,
                      self.win_length, self.window, self.normalized)
        sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)
        return sc_loss, mag_loss

class STFTLoss(nn.Module):
    def __init__(self,
                 n_fft: int = 1024,
                 hop_length: int = 120,
                 win_length: int = 600,
                 window: str = "hann_window",
                 normalized: bool = False,
                 factor_sc: float = 0.1,
                 factor_mag: float = 0.1,
                 epsilon: float = torch.finfo(torch.float32).eps):
        super().__init__()
        self.loss = STFTLosses(n_fft, hop_length, win_length, window, normalized, epsilon)
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        sc_loss, mag_loss = self.loss(x, y)
        return self.factor_sc * sc_loss + self.factor_mag * mag_loss

class MRSTFTLoss(nn.Module):
    def __init__(self,
                 n_ffts: Optional[Sequence[int]] = None,
                 hop_lengths: Optional[Sequence[int]] = None,
                 win_lengths: Optional[Sequence[int]] = None,
                 window: str = "hann_window",
                 factor_sc: float = 0.1,
                 factor_mag: float = 0.1,
                 normalized: bool = False,
                 epsilon: float = torch.finfo(torch.float32).eps):
        super().__init__()
        n_ffts = n_ffts or [1024, 2048, 512]
        hop_lengths = hop_lengths or [120, 240, 50]
        win_lengths = win_lengths or [600, 1200, 240]
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.stft_losses = nn.ModuleList()
        for fs, ss, wl in zip(n_ffts, hop_lengths, win_lengths):
            self.stft_losses.append(STFTLosses(fs, ss, wl, window, normalized, epsilon))
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        sc_loss = torch.Tensor([0.0]).to(x.device)
        mag_loss = torch.Tensor([0.0]).to(x.device)
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)
        loss = self.factor_sc * sc_loss + self.factor_mag * mag_loss
        loss = loss.squeeze(0)
        return loss