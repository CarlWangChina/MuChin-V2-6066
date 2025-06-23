import torch
import torch.nn as nn
from music_dit.utils import get_logger
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.models.music_dit import MusicDiTModel

logger = get_logger(__name__)

class MusicDiTGenerator(nn.Module):
    def __init__(self):
        super(MusicDiTGenerator, self).__init__()
        self.model = MusicDiTModel()