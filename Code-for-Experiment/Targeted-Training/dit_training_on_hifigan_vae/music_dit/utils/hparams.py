from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from music_dit.utils import get_logger, probe_device
import Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.dist_wrapper as dist

logger = get_logger(__name__)
_configs = OmegaConf.create()

def get_hparams() -> DictConfig:
    global _configs
    if "root_dir" not in _configs:
        root_dir = Path(__file__).parent.parent.parent
        _configs = OmegaConf.create()
        _configs.root_dir = root_dir
        _configs.configs_dir = root_dir / "music_dit" / "configs"
        _configs.checkpoint_dir = root_dir / "checkpoints"
        _configs.log_dir = root_dir / "logs"
        _configs.data_dir = root_dir / "data"
        _configs.device = str(dist.get_device())
        if _configs.configs_dir.exists() and _configs.configs_dir.is_dir():
            for file in _configs.configs_dir.glob("*.yaml"):
                with open(file, "r") as f:
                    user_configs = OmegaConf.load(f)
                    _configs = OmegaConf.merge(_configs, user_configs)
                    if dist.is_primary():
                        logger.info("Loaded user configurations from %s", file)
    return _configs