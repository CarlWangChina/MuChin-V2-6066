import torch
import Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.dist_wrapper as dist

def probe_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{dist.get_local_rank()}")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")