from pathlib import Path
from typing import Any
import deepspeed
import torch
from ama_prof_divi_common.utils import get_hparams, setup_random_seed
from ..model import VAE2Model
from Code_for_Experiment.Targeted_Training.hifigan_vocoder.ama-prof-divi_hifigan.model.discriminators import Discriminator
from .dataset import build_vae_dataset
from .trainer import VAE2Trainer

logger = deepspeed.logger

def build_vae2_trainer(cmd_args: Any) -> VAE2Trainer:
    root_dir = Path(__file__).parent.parent.parent
    hparams = get_hparams(root_dir)
    logger.info("Building the VAE2 trainer...")
    torch.backends.cudnn.enabled = hparams.training.trainer.cudnn_enabled
    setup_random_seed(hparams.training.trainer.random_seed)
    logger.info(f"Building models...")
    model_g = VAE2Model(training=True)
    model_d = Discriminator()
    logger.info(f"Initializing DeepSpeed engine...")
    model_engine_g, _, _, lr_scheduler_g = deepspeed.initialize(
        args=cmd_args,
        model=model_g
    )
    model_engine_d, _, _, lr_scheduler_d = deepspeed.initialize(
        args=cmd_args,
        model=model_d,
        dist_init_required=False
    )
    logger.info(f"Build datasets...")
    dataset = build_vae_dataset()
    return VAE2Trainer(
        dataset=dataset,
        model_engine_g=model_engine_g,
        lr_scheduler_g=lr_scheduler_g,
        model_engine_d=model_engine_d,
        lr_scheduler_d=lr_scheduler_d
    )