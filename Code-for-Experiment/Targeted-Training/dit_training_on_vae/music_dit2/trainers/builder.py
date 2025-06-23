import torch
import deepspeed
from typing import Any
from pathlib import Path
from .trainer import MusicDiTTrainer
import MusicDiT.Trainer
import get_hparams, setup_random_seed
from ama_prof_divi_common.utils.dist_wrapper import get_rank
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.models.music_dit import MusicDiTModel
logger = deepspeed.logger

def build_music_dit_trainer(cmd_args: Any) -> MusicDiTTrainer:
    rank = get_rank()
    logger.info("Building Music-DIT trainer for rank %d", rank)
    root_path = Path(__file__).parent.parent.parent
    hparams = get_hparams(root_path)
    torch.backends.cudnn.enabled = hparams.training.trainer.cudnn_enabled
    setup_random_seed(hparams.training.trainer.random_seed)
    model = MusicDiTModel()
    model_engine, _, _, lr_scheduler = deepspeed.initialize(
        args=cmd_args,
        model=model
    )
    return MusicDiTTrainer(model_engine=model_engine, lr_scheduler=lr_scheduler)