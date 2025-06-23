from pathlib import Path
from omegaconf import OmegaConf
from .generator import Generator
from Code_for_Experiment.Targeted_Training.hifigan_vocoder.ama-prof-divi_hifigan.model.discriminators import Discriminator

def build_generator(version: str = "v1"):
    config_file = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"Config file ({config_file}) not found.")
    with open(config_file, "r") as f:
        hparams = OmegaConf.load(f)
    if version not in hparams.hifigan:
        raise ValueError(f"Bad config file, or invalid version: {version}")
    n_mels = hparams.mel_default.n_mels
    cfg = hparams.hifigan[version]

    return Generator(hparams=hparams,
                     version=version,
                     upsampling_rates=cfg.upsampling_rates,
                     upsampling_kernel_sizes=cfg.upsampling_kernel_sizes,
                     upsampling_initial_channel=cfg.upsampling_initial_channel,
                     resblock_kernel_sizes=cfg.resblock_kernel_sizes,
                     resblock_dilation_sizes=cfg.resblock_dilation_sizes,
                     resblock=cfg.resblock,
                     n_mels=n_mels)

def build_discriminator():
    return Discriminator()