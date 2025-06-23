import sys
from pathlib import Path
from omegaconf import OmegaConf
import torch

current_path = Path(__file__).absolute()
root_path = current_path.parent.parent.parent
sys.path.append(str(root_path))
sys.path.append(str(current_path))

from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger
import TrainerArgs, TrainingPhaseArgs, start_trainer

logger = get_logger(__name__)
CONFIG_FILE = "config.yaml"

_configs = OmegaConf.load(current_path / CONFIG_FILE)
_configs = _configs.acoustic_trainer

trainer_args = TrainerArgs(
    name="acoustic_embedding",
    data_dir=_configs.diffusion.data_dir,
    checkpoint_dir=_configs.embedding.checkpoint_dir,
    log_dir=_configs.embedding.log_dir,
    data_file_pattern=_configs.diffusion.data_file_pattern,
    parallel_backend=_configs.parallel.backend,
    master_addr=_configs.parallel.master_addr,
    master_port=_configs.parallel.master_port,
    use_data_files_proportion=_configs.embedding.use_data_files_proportion,
    training_dataset_proportion=_configs.embedding.training_dataset_proportion,
    extra_args={
        "dim": _configs.embedding.dim,
        "window_size": _configs.diffusion.window_size,
        "vocab_size": _configs.diffusion.vocab_size,
        "start_id": _configs.diffusion.start_id,
        "pad_id": _configs.diffusion.pad_id,
        "clap_dim": _configs.diffusion.clap_dim,
        "tokens_chunk_len": _configs.diffusion.tokens_chunk_len,
        "tokens_num_q": _configs.diffusion.tokens_num_q,
    },
)

trainer_args.validation_batch_size = _configs.embedding.batch_size
trainer_args.shuffle = _configs.embedding.shuffle
trainer_args.random_seed = _configs.embedding.random_seed
trainer_args.phases = [
    TrainingPhaseArgs(
        name="embedding_%d" % _configs.embedding.dim,
        optimizer=_configs.embedding.phases.embedding.optimizer,
        learning_rate=_configs.embedding.phases.embedding.learning_rate,
        max_epochs=_configs.embedding.phases.embedding.epochs,
        batch_size=_configs.embedding.batch_size,
        ckpt_interval=_configs.embedding.ckpt_interval,
    )
]

if __name__ == "__main__":
    if not torch.cuda.is_available():
        trainer_args.parallel_enabled = False
    trainer_class_name = "ama-prof-divi.training.trainers.acoustic.EmbeddingTrainer"
    start_trainer(trainer_class_name, trainer_args)