import sys
from pathlib import Path
from omegaconf import OmegaConf
import torch

current_path = Path(__file__).absolute()
root_path = current_path.parent.parent.parent
sys.path.append(str(root_path))
sys.path.append(str(current_path))
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger
import TrainerArgs
logger = get_logger(__name__)
CONFIG_FILE = "config.yaml"
_configs = OmegaConf.load(current_path / CONFIG_FILE)
_configs = _configs.mert_rvq_trainer
trainer_args = TrainerArgs(
    name="mert_rvq",
    data_dir=_configs.data_dir,
    checkpoint_dir=_configs.checkpoint_dir,
    log_dir=_configs.log_dir,
    data_file_pattern=_configs.data_file_pattern,
    parallel_backend=_configs.parallel_backend,
    master_addr=_configs.master_addr,
    master_port=_configs.master_port,
    use_data_files_proportion=_configs.use_data_files_proportion,
    training_dataset_proportion=_configs.training_dataset_proportion,
    extra_args={
        "window_size": _configs.window_size,
        "feature_dim": _configs.feature_dim,
        "feature_rate": _configs.feature_rate,
        "stride": _configs.stride,
        "num_quantizers": _configs.num_quantizers,
        "codebook_size": _configs.codebook_size,
        "similarity": _configs.similarity
    }
)
trainer_args.phases[0].max_epochs = _configs.training_epochs

if __name__ == "__main__":
    if not torch.cuda.is_available():
        trainer_args.parallel_enabled = False
    trainer_class_name = "ama_prof_divi.training.trainers.mert_rvq.MertRVQTrainer"
    start_trainer = TrainerArgs.start_trainer
    start_trainer(trainer_class_name, trainer_args)