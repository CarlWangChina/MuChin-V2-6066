import torch
from typing import Optional
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.hparams import get_hparams
from .acoustic_generator import AccompanimentGenerator

_accompaniment_generator = None

def get_accompaniment_generator(hparams: dict = None, device: Optional[torch.device or str] = None) -> AccompanimentGenerator:
    global _accompaniment_generator
    if _accompaniment_generator is None:
        if hparams is None:
            hparams = get_hparams()
        _accompaniment_generator = AccompanimentGenerator(hparams, device=device)
    return _accompaniment_generator