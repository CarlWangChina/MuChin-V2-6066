import sys
import ray
import argparse
import deepspeed
from pathlib import Path

current_path = Path(__file__).absolute()
root_path = current_path.parent.parent
sys.path.append(str(root_path))
from ama_prof_divi_vae2.trainer import build_vae2_trainer

ray.init("auto")

parser = argparse.ArgumentParser(description="Training script for VAE2.")
parser = deepspeed.add_config_arguments(parser)
parser.add_argument("--local_rank", type=int, default=-1, help="Local rank")
parser.set_defaults(deepspeed_config=str(root_path / "configs/deepspeed_config.json"))

cmd_args = parser.parse_args()
trainer = build_vae2_trainer(cmd_args)
trainer.start_training()
trainer.close()