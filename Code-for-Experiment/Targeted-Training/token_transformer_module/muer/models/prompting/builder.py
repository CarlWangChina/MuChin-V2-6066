import torch
from typing import Optional
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.hparams import get_hparams
import PromptEncoder
from .laion_clap_prompt_encoder import LaionClapPromptEncoder

_prompt_encoder = None

def get_prompt_encoder(hparams: Optional[dict] = None, device: Optional[str or torch.device] = None) -> PromptEncoder:
    global _prompt_encoder
    if _prompt_encoder is None:
        if hparams is None:
            hparams = get_hparams()
        name = hparams["ama-prof-divi"]["models"]["prompting"]["encoder"]["name"]
        if name == "laion_clap":
            _prompt_encoder = LaionClapPromptEncoder(hparams, device=device)
        else:
            raise ValueError(f"Unknown prompt encoder: {name}")
    return _prompt_encoder