import torch.nn as nn
import deepspeed
from typing import Union
from pathlib import Path
from ama_prof_divi_common.utils import get_hparams
from .kld_loss import KLDivergenceLoss
from .spec_loss import MultiScaleMelSpectrogramLoss
from .stft_loss import MRSTFTLoss
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.modules.vae.training.losses.sisnrloss import SISNRLoss

root_path = Path(__file__).parent.parent.parent.parent
hparams = get_hparams(root_path)

def get_kl_loss(reduction: str = 'mean') -> nn.Module:
    assert reduction in ['mean', 'sum'], f'Unsupported reduction method: {reduction}'
    return KLDivergenceLoss(reduction=reduction)

def get_l1_loss() -> nn.Module:
    return nn.L1Loss()

def get_l2_loss() -> nn.Module:
    return nn.MSELoss()

def get_mel_loss() -> nn.Module:
    return MultiScaleMelSpectrogramLoss(
        sampling_rate=hparams.vae.sampling_rate,
        range_start=hparams.training.msspec.range_start,
        range_end=hparams.training.msspec.range_end,
        n_mels=hparams.training.msspec.n_mels,
        f_min=hparams.training.msspec.f_min,
        f_max=hparams.training.msspec.f_max,
        normalized=hparams.training.msspec.normalized,
        alphas=hparams.training.msspec.alphas,
        floor_level=hparams.training.msspec.floor_level
    )

def get_stft_loss() -> nn.Module:
    return MRSTFTLoss()

def get_sisnr_loss() -> nn.Module:
    return SISNRLoss(sample_rate=hparams.vae.sampling_rate)