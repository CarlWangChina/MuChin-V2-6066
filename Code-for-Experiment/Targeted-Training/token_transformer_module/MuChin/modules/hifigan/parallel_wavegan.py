import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SineGen(nn.Module):
    def __init__(self, *, sampling_rate: int, harmonic_num: int = 0, sine_amp: float = 0.1, noise_std: float = 0.003, voiced_threshold: float = 0.0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = sampling_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0: torch.Tensor) -> torch.Tensor:
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0_values: torch.Tensor, upp: float) -> torch.Tensor:
        rad_values = (f0_values / self.sampling_rate).fmod(1.)
        rand_ini = torch.rand(1, self.dim, device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] += rand_ini
        is_half = rad_values.dtype is not torch.float32
        tmp_over_one = torch.cumsum(rad_values.double(), 1)
        if is_half:
            tmp_over_one = tmp_over_one.half()
        else:
            tmp_over_one = tmp_over_one.float()
        tmp_over_one *= upp
        tmp_over_one = F.interpolate(tmp_over_one.transpose(2, 1), scale_factor=upp, mode='linear', align_corners=True).transpose(2, 1)
        rad_values = F.interpolate(rad_values.transpose(2, 1), scale_factor=upp, mode='nearest').transpose(2, 1)
        tmp_over_one = tmp_over_one.fmod(1.)
        diff = F.conv2d(tmp_over_one.unsqueeze(1), torch.FloatTensor([[[[-1.], [1.]]]]).to(tmp_over_one.device), stride=(1, 1), padding=0, dilation=(1, 1)).squeeze(1)
        cum_sum_shift = (diff < 0).double()
        cum_sum_shift = torch.cat((torch.zeros((1, 1, self.dim), dtype=torch.double).to(f0_values.device), cum_sum_shift), dim=1)
        sines = torch.sin(torch.cumsum(rad_values.double() + cum_sum_shift, dim=1) * 2 * np.pi)
        if is_half:
            sines = sines.half()
        else:
            sines = sines.float()
        return sines

    def forward(self, f0: torch.Tensor, *, upp: float) -> (torch.Tensor, torch.Tensor):
        f0 = f0.unsqueeze(-1)
        fn = torch.multiply(f0, torch.arange(1, self.dim + 1, device=f0.device).reshape((1, 1, -1)))
        sine_waves = self._f02sine(fn, upp) * self.sine_amp
        uv = (f0 > self.voiced_threshold).float()
        uv = F.interpolate(uv.transpose(2, 1), scale_factor=upp, mode='nearest').transpose(2, 1)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv

class SourceModuleHnNSF(nn.Module):
    def __init__(self, sampling_rate: int, harmonic_num: int = 0, sine_amp: float = 0.1, noise_std: float = 0.003, voiced_threshold: float = 0.0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.l_sin_gen = SineGen(sampling_rate=sampling_rate, harmonic_num=harmonic_num, sine_amp=sine_amp, noise_std=noise_std, voiced_threshold=voiced_threshold)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x, upp):
        sine_wave = self.l_sin_gen(x, upp)
        sine_merge = self.l_tanh(self.l_linear(sine_wave))
        return sine_merge