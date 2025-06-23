import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from einops import rearrange
from librosa import hz_to_midi
import logging
from .predictors import (DurationPredictor, PitchPredictor, EnergyPredictor)
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.modules.fastspeech2.lr import LengthRegulator
ENERGY_LEVELS = 256
logger = logging.getLogger(__name__)

class FS2Variance(nn.Module):
    def __init__(self, hidden_size: int, duration_predictor_layers: int, duration_predictor_kernel_size: int, pitch_predictor_layers: int, pitch_predictor_kernel_size: int, energy_predictor_layers: int, energy_predictor_kernel_size: int, dropout: float = 0.1, num_speakers: int = 1, padding_type: str = "same", pitch_max: int = 128, rate_ratio: float = 1.0, enable_energy: bool = True, device: str = "cpu"):
        super(FS2Variance, self).__init__()
        self.rate_ratio = rate_ratio
        self.enable_energy = enable_energy
        self.device = device
        self.pitch_embedding = nn.Embedding(pitch_max, hidden_size, device=device)
        if num_speakers > 1:
            self.spk_pitch_embedding = nn.Embedding(num_speakers, hidden_size, device=device)
        else:
            self.spk_pitch_embedding = None
        self.energy_embedding = nn.Embedding(ENERGY_LEVELS, hidden_size, device=device)
        self.duration_predictor = DurationPredictor(dim=hidden_size, num_layers=duration_predictor_layers, kernel_size=duration_predictor_kernel_size, dropout=dropout, padding_type=padding_type, device=device)
        self.pitch_predictor = PitchPredictor(dim=hidden_size, num_layers=pitch_predictor_layers, kernel_size=pitch_predictor_kernel_size, dropout=dropout, padding_type=padding_type, device=device)
        self.energy_predictor = EnergyPredictor(dim=hidden_size, num_layers=energy_predictor_layers, kernel_size=energy_predictor_kernel_size, dropout=dropout, padding_type=padding_type, device=device)
        self.length_regulator = LengthRegulator()

    def forward(self, phoneme: torch.Tensor, phoneme_duration: Optional[torch.Tensor] = None, pitch_tokens: Optional[torch.Tensor] = None, speaker_ids: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> (torch.Tensor, torch.Tensor):
        if speaker_ids is not None:
            assert self.spk_pitch_embedding is not None, "The speaker embedding layer is not initialized because the hyper param num_speakers = 1."
            phoneme = phoneme + self.spk_pitch_embedding(speaker_ids)
        if phoneme_duration is None:
            phoneme_duration, _ = self.duration_predictor(phoneme, mask=mask)
        length_idx, dur_cum_sum = self.length_regulator(phoneme_duration, padding_mask=mask, rate_ratio=self.rate_ratio)
        dur_cum_sum = dur_cum_sum[:, -1].to(self.device)
        max_len = dur_cum_sum.max().item()
        length_idx = rearrange(length_idx.to(self.device), "b t -> b t ()")
        length_idx = length_idx.expand(-1, -1, phoneme.shape[-1])
        phoneme = F.pad(phoneme, (0, 0, 1, 0))
        h = torch.gather(phoneme, 1, length_idx)
        assert h.shape == (phoneme.shape[0], max_len, phoneme.shape[-1])
        logger.info(f"Expanded phoneme: from {phoneme.shape} to {h.shape}")
        mask = torch.zeros((phoneme.shape[0], max_len), device=self.device)
        for batch in range(mask.shape[0]):
            mask[batch, :dur_cum_sum[batch]] = 1.0
        del phoneme
        if pitch_tokens is not None:
            assert pitch_tokens.dim() == 2, f"Pitch tokens should be in 2D tensor, but got {pitch_tokens.dim()}D tensor."
            assert pitch_tokens.shape[0] == h.shape[0], f"Batch size mismatched between pitch tokens ({pitch_tokens.shape[0]}) and input ({h.shape[0]})."
            assert pitch_tokens.shape[1] >= h.shape[1], (f"Melody length ({pitch_tokens.shape[1]}) should be greater than or equal to the phoneme length ({h.shape[1]}), ")
            if pitch_tokens.shape[1] > h.shape[1]:
                logger.warning(f"Truncated pitch tokens to match the phoneme length.  From {pitch_tokens.shape[1]} to {h.shape[1]}.")
                pitch_tokens = pitch_tokens[:, :h.shape[1]]
            pitch_duration = torch.ones(pitch_tokens.shape).to(self.device)
            length_idx_melody, _ = self.length_regulator(pitch_duration, rate_ratio=self.rate_ratio)
            length_idx_melody = rearrange(length_idx_melody.to(self.device), "b t -> b t ()")
            pitch_tokens = F.pad(pitch_tokens.unsqueeze(-1), (0, 0, 1, 0))
            pitch_tokens = torch.gather(pitch_tokens, 1, length_idx_melody)
            pitch_tokens.squeeze_(-1)
            logger.info("Expanded pitch tokens: from {} to {}".format(pitch_duration.shape, pitch_tokens.shape))
            if pitch_tokens.shape[1] < h.shape[1]:
                pitch_tokens = F.pad(pitch_tokens, (0, 0, 0, h.shape[1] - pitch_tokens.shape[1]))
            elif pitch_tokens.shape[1] > h.shape[1]:
                pitch_tokens = pitch_tokens[:, :h.shape[1]]
        else:
            f0 = self.pitch_predictor(h)[..., 0]
            f0 = torch.clamp(f0, min=80.0, max=10000.0).cpu()
            pitch_tokens = torch.tensor(hz_to_midi(f0)).long().unsqueeze(-1).to(self.device)
        pitch_tokens = pitch_tokens.squeeze(-1)
        h = h + self.pitch_embedding(pitch_tokens)
        if self.enable_energy:
            energy = self.energy_predictor(h)[..., 0] * ENERGY_LEVELS + 0.5
            energy = torch.clamp(energy, min=0.0, max=ENERGY_LEVELS - 1).long()
            h = h + self.energy_embedding(energy)
        return h, mask