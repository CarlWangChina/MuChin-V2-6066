from pathlib import Path
from omegaconf import OmegaConf
import MelGenerator

def build_mel_generator():
    config_file = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"Config file ({config_file}) not found.")
    with open(config_file, "r") as f:
        hparams = OmegaConf.load(f)
    mel_cfg = hparams.mel_default
    mel_generator = MelGenerator(**mel_cfg)
    return mel_generator