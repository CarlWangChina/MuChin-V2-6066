import torch
import deepspeed
from typing import Any
from pathlib import Path
from ama_prof_divi_common.utils import get_hparams, setup_random_seed
from ama_prof_divi_common.utils.dist_wrapper import get_rank
from ..model import VAE2Model
from .trainer import VAE2Trainer
from Code_for_Experiment.Targeted_Training.hifigan_vocoder.ama-prof-divi_hifigan.model.discriminators import Discriminator

logger = deepspeed.logger

def build_vae2_trainer(cmd_args: Any) -> VAE2Trainer:
    root_dir = Path(__file__).parent.parent.parent
    hparams = get_hparams(root_dir)
    rank = get_rank()
    if rank == 0:
        logger.info("Building the VAE2 trainer...")
    torch.backends.cudnn.enabled = hparams.training.trainer.cudnn_enabled
    setup_random_seed(hparams.training.trainer.random_seed)
    model_g = VAE2Model(training=True)
    model_d = Discriminator()
    model_engine_g, _, _, lr_scheduler_g = deepspeed.initialize(
        args=cmd_args,
        model=model_g
    )
    model_engine_d, _, _, lr_scheduler_d = deepspeed.initialize(
        args=cmd_args,
        model=model_d,
        dist_init_required=False
    )
    return VAE2Trainer(
        model_engine_g=model_engine_g,
        lr_scheduler_g=lr_scheduler_g,
        model_engine_d=model_engine_d,
        lr_scheduler_d=lr_scheduler_d
    )