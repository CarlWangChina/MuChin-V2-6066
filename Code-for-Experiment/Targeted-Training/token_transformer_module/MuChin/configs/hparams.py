import os
import torch
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import logging
from ama_prof_divi.utils import probe_devices

logger = logging.getLogger(__name__)
_configs = OmegaConf.create()

def init_hparams(config_file="ama-prof-divi-config.yaml"):
    global _configs
    _configs = OmegaConf.create()
    root_path = Path(__file__).parent.parent.parent
    config_path = root_path.joinpath("configs").joinpath(config_file)
    logger.info(f"Loading hyper-parameters from {config_path}")
    if config_path.exists():
        _configs = OmegaConf.load(config_path)
        assert isinstance(_configs, DictConfig)
    else:
        logger.error(f"Hyper-parameters not found in {config_path}")
        _configs = OmegaConf.create()
    if "ama-prof-divi" not in _configs:
        _configs.ama_prof_divi = OmegaConf.create()
    _configs.ama_prof_divi.root_path = root_path
    if "models" not in _configs.ama_prof_divi:
        _configs.ama_prof_divi.models = OmegaConf.create()
    models_path = root_path.joinpath("ama-prof-divi").joinpath("models")
    for model_path in models_path.iterdir():
        if model_path.is_dir():
            model_name = model_path.name
            model_config_path = model_path.joinpath("configs").joinpath("config.yaml")
            if model_config_path.exists():
                logger.info(f"Loading hyper-parameters for the '{model_name}' model, from {model_config_path}")
                model_configs = OmegaConf.load(model_config_path)
                assert isinstance(model_configs, DictConfig)
                _configs.ama_prof_divi.models = OmegaConf.merge(_configs.ama_prof_divi.models, model_configs)
    trainers_cfg_dir = root_path.joinpath("ama-prof-divi").joinpath("training", "configs")
    if trainers_cfg_dir.exists() and trainers_cfg_dir.is_dir():
        trainer_config_file = trainers_cfg_dir.joinpath("config.yaml")
        if trainer_config_file.exists():
            logger.info(f"Loading hyper-parameters for training, from {trainer_config_file}")
            trainer_configs = OmegaConf.load(trainer_config_file)
            assert isinstance(trainer_configs, DictConfig)
            _configs.ama_prof_divi.trainers = trainer_configs

def get_hparams() -> dict:
    hparams = OmegaConf.to_container(_configs, resolve=True)
    if isinstance(hparams["ama-prof-divi"]["device"], str):
        hparams["ama-prof-divi"]["device"] = torch.device(hparams["ama-prof-divi"]["device"])
    return hparams

def merge_hparams(hparams: DictConfig or dict):
    global _configs
    _configs = OmegaConf.merge(_configs, hparams)

def _merge_hparams_with_env():
    global _configs
    _configs.env = OmegaConf.create()
    for key, value in os.environ.items():
        key = key.upper()
        _configs.env[key] = value

def _merge_hparams_with_cli():
    global _configs
    _configs.cli = OmegaConf.from_cli()

def post_init_hparams():
    _merge_hparams_with_env()
    _merge_hparams_with_cli()
    device, device_ids = probe_devices()
    if "device" not in _configs.ama_prof_divi:
        _configs.ama_prof_divi.device = device
    if "device_ids" not in _configs.ama_prof_divi:
        _configs.ama_prof_divi.device_ids = device_ids