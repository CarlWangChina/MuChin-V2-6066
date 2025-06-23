import torch
import ama_prof_divi_vae2.model.vae as vae
from Code_for_Experiment.Targeted_Training.hifigan_vocoder.ama-prof-divi_hifigan.model.discriminators import Discriminator

print(sum(p.numel() for p in Discriminator().parameters()))