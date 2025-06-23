import sys
import argparse
import deepspeed
from pathlib import Path
import ama_prof_divi_common.utils.dist_wrapper as dist
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.random import setup_random_seed

current_path = Path(__file__).absolute()
root_path = current_path.parent.parent
sys.path.append(str(root_path))

from ama_prof_divi_hifigan.trainer import build_hifigan_trainer

parser = argparse.ArgumentParser(description="Training script for Hifi-GAN.")
parser = deepspeed.add_config_arguments(parser)
parser.add_argument("--local_rank", type=int, default=-1, help="Local rank")
parser.set_defaults(deepspeed_config=str(root_path / "configs/deepspeed_config.json"))
cmd_args = parser.parse_args()
trainer = build_hifigan_trainer(cmd_args)
trainer.start_training()
trainer.close()