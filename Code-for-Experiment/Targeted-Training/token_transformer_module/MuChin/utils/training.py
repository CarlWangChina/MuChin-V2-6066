import torch.nn as nn
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.hparams import get_hparams

def model_need_freeze(model_name: str):
    hparams = get_hparams()
    if model_name in hparams["ama-prof-divi"]["training"]["frozen_models"]:
        return True
    else:
        return False

def freeze_model(model: nn.Module, freeze: bool = True):
    for param in model.parameters():
        param.requires_grad = not freeze