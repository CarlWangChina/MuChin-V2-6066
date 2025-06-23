import torch
import deepspeed
from typing import Any
from pathlib import Path
from .trainer import HifiGANTrainer
import HifiGAN.Trainer
import get_hparams, setup_random_seed
from ama_prof_divi_common.utils.dist_wrapper import get_rank
from ama_prof_divi_hifigan.model import build_generator, build_discriminator
from Code_for_Experiment.Targeted_Training.hifigan_vocoder.ama-prof-divi_hifigan.mel.mel import get_mel_generator
logger = deepspeed.logger

def build_hifigan_trainer(cmd_args: Any) -> HifiGANTrainer:
    rank = get_rank()
    logger.info("Building HifiGAN trainer for rank %d", rank)
    root_path = Path(__file__).parent.parent.parent
    hparams = get_hparams(root_path)
    torch.backends.cudnn.enabled = hparams.training.trainer.cudnn_enabled
    setup_random_seed(hparams.training.trainer.random_seed)
    model_g = build_generator(hparams.training.trainer.version)
    model_d = build_discriminator()
    mel_generator = get_mel_generator(model_g)
    model_engine_g, _, _, lr_scheduler_g = deepspeed.initialize(
        args=cmd_args,
        model=model_g
    )
    model_engine_d, _, _, lr_scheduler_d = deepspeed.initialize(
        args=cmd_args,
        model=model_d
    )
    return HifiGANTrainer(
        model_engine_g=model_engine_g,
        lr_scheduler_g=lr_scheduler_g,
        model_engine_d=model_engine_d,
        lr_scheduler_d=lr_scheduler_d,
        mel_generator=mel_generator
    )