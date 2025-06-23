import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger
logger = get_logger(__name__)

def _dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

def _dynamic_range_compression_torch(x, C=1, clip_val=1e-7):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def _spectral_normalize_torch(magnitudes):
    output = _dynamic_range_compression_torch(magnitudes)
    return output

class MelEncoder(nn.Module):
    def __init__(self, hparams: dict, device: str):
        super(MelEncoder, self).__init__()
        self.device = device
        self.n_fft = hparams["n_fft"]
        self.num_mel_bands = hparams["num_mels"]
        self.sampling_rate = hparams["sampling_rate"]
        self.hop_size = hparams["hop_size"]
        self.win_size = hparams["win_size"]
        self.f_min = hparams["fmin"]
        self.f_max = hparams["fmax"]
        self.mel = librosa_mel_fn(sr=self.sampling_rate, n_fft=self.n_fft, n_mels=self.num_mel_bands, fmin=self.f_min, fmax=self.f_max)
        self.mel_basis = torch.from_numpy(self.mel).to(self.device)
        self.hann_window = torch.hann_window(self.win_size).to(self.device)

    def forward(self, x: torch.Tensor, center: bool = False) -> torch.Tensor:
        x = F.pad(x.unsqueeze(1), pad=[(self.n_fft - self.hop_size) // 2, (self.n_fft - self.hop_size) // 2], mode="reflect")
        x = x.squeeze(1)
        spec = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_size, win_length=self.win_size, window=self.hann_window, center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
        spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-7)
        spec = torch.matmul(self.mel_basis, spec)
        spec = _spectral_normalize_torch(spec)
        return spec