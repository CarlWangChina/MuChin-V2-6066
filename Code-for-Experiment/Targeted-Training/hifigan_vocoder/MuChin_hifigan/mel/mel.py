import torch
import torch.nn as nn
import torchaudio
from typing import Optional
from Code_for_Experiment.Targeted_Training.hifigan_vocoder.ama-prof-divi_hifigan.model.generator import Generator

class MelGenerator(nn.Module):
    def __init__(self, sampling_rate: int = 24000, n_fft: int = 1024, win_length: Optional[int] = None, hop_length: Optional[int] = None, n_mels: int = 80, f_min: Optional[float] = None, f_max: Optional[float] = None, min_value: float = 1e-5, scale: float = 0.1):
        super(MelGenerator, self).__init__()
        win_length = win_length or n_fft
        self.sampling_rate = sampling_rate
        self.n_mels = n_mels
        self.hop_length = hop_length or win_length // 2
        self.min_value = min_value
        self.scale = scale
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sampling_rate, n_fft=n_fft, win_length=win_length, hop_length=self.hop_length, n_mels=n_mels, f_min=f_min, f_max=f_max)

    def get_mel_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mel_transform(x)
        if x.size(-1) % self.hop_length == 0:
            y = y[..., :x.size(-1) // self.hop_length]
        y = torch.log(torch.clamp(y, min=self.min_value) * self.scale)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_mel_spectrogram(x)

def get_mel_generator(generator: Generator):
    cfg = generator.hparams.mel_default
    mel_generator = MelGenerator(sampling_rate=cfg.sampling_rate, n_fft=cfg.n_fft, win_length=cfg.win_length, hop_length=cfg.hop_length, n_mels=cfg.n_mels, f_min=cfg.f_min, f_max=cfg.f_max, min_value=cfg.min_value, scale=cfg.scale)
    return mel_generator