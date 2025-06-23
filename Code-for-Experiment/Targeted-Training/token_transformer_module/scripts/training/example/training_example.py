import sys
from pathlib import Path
import torch

current_path = Path(__file__).absolute()
root_path = current_path.parent.parent.parent
sys.path.append(str(root_path))
sys.path.append(str(current_path))

from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger
from ama_prof_divi.training import Trainer, TrainerArgs

logger = get_logger(__name__)

_trainer_args = TrainerArgs(
    name="example",
    data_dir=str(root_path / "data" / "example_trainer"),
    checkpoint_dir=str(root_path / "data" / "example_trainer" / "ckpts"),
    log_dir=str(root_path / "data" / "example_trainer" / "logs"),
    data_file_pattern=r"^[\S]+\.csv$",
    parallel_backend="nccl",
)
_trainer_args.phases[0].max_epochs = 5
_trainer_args.validation_batch_size = 10

if __name__ == "__main__":
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        _trainer_args.parallel_enabled = False
    trainer_class_name = "ama_prof_divi.training.trainers.example.ExampleTrainer"
    start_trainer = Trainer.start_trainer
    start_trainer(trainer_class_name, _trainer_args)