import sys
import torch
from pathlib import Path
import argparse
import deepspeed
import deepspeed.comm as dist

current_path = Path(__file__).absolute()
root_path = current_path.parent.parent
sys.path.append(str(root_path))

from music_dit.utils import setup_random_seed, get_hparams
from music_dit.modules.vae.training import VAEModelForTraining, VAETrainer, VAETrainingDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for pitch re-generation")
    parser = deepspeed.add_config_arguments(parser)
    hparams = get_hparams()
    torch.backends.cudnn.enabled = False
    setup_random_seed(hparams.training.vae.random_seed)
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank")
    parser.set_defaults(deepspeed_config=str(hparams.configs_dir / "deepspeed_config_vae.json"))
    cmd_args = parser.parse_args()
    model = VAEModelForTraining()
    deepspeed.logger.info("Initializing the deepspeed engine.")
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(args=cmd_args, model=model)
    dist.barrier()
    rank = dist.get_rank()
    if rank == 0:
        deepspeed.logger.info("DeepSpeed engine initialized.")
        deepspeed.logger.info(f"Initializing the random seed.")
    if rank == 0:
        deepspeed.logger.info("Loading the dataset, please wait...")
    training_dataset, validation_dataset = VAETrainingDataset.load_datasets(
        hparams.training.vae.use_data_files_proportion,
        hparams.training.vae.training_dataset_proportion
    )
    dist.barrier()
    if rank == 0:
        deepspeed.logger.info(f"Dataset loaded.  Training dataset: {len(training_dataset)}, Validation dataset: {len(validation_dataset)}")
    dist.barrier()
    trainer = VAETrainer(model_engine=model_engine, optimizer=optimizer, lr_scheduler=lr_scheduler, training_dataset=training_dataset, validation_dataset=validation_dataset)
    dist.barrier()
    trainer.load_checkpoints()
    trainer.train()