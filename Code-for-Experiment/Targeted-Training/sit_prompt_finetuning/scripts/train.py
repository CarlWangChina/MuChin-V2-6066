import sys
import argparse
import deepspeed
from pathlib import Path
import ama_prof_divi_common.utils.dist_wrapper as dist
import torch.multiprocessing as mp
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.random import setup_random_seed

current_path = Path(__file__).absolute()
root_path = current_path.parent.parent
sys.path.append(str(root_path))

from music_dit2.trainers import build_music_dit_trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for Music-DIT.")
    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank")
    parser.set_defaults(deepspeed_config=str(root_path / "configs/deepspeed_config.json"))
    cmd_args = parser.parse_args()
    trainer = build_music_dit_trainer(cmd_args)
    mp.set_start_method('spawn')
    trainer.start_training()
    trainer.close()