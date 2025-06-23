import sys
from pathlib import Path
from omegaconf import OmegaConf
import torch

current_path = Path(__file__).absolute()
root_path = current_path.parent.parent.parent
sys.path.append(str(root_path))
sys.path.append(str(current_path))
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger
import Trainer
from Trainer import TrainerArgs, TrainingPhaseArgs, start_trainer

logger = get_logger(__name__)
CONFIG_FILE = "config.yaml"
_configs = OmegaConf.load(current_path / CONFIG_FILE)
_configs = _configs.acoustic_trainer
trainer_args = TrainerArgs(
    name="acoustic_diffusion",
    data_dir=_configs.diffusion.data_dir,
    checkpoint_dir=_configs.diffusion.checkpoint_dir,
    log_dir=_configs.diffusion.log_dir,
    data_file_pattern=_configs.diffusion.data_file_pattern,
    parallel_backend=_configs.parallel.backend,
    master_addr=_configs.parallel.master_addr,
    master_port=_configs.parallel.master_port,
    use_data_files_proportion=_configs.diffusion.use_data_files_proportion,
    training_dataset_proportion=_configs.diffusion.training_dataset_proportion,
    extra_args={
        "window_size": _configs.diffusion.window_size,
        "vocab_size": _configs.diffusion.vocab_size,
        "start_id": _configs.diffusion.start_id,
        "pad_id": _configs.diffusion.pad_id,
        "clap_dim": _configs.diffusion.clap_dim,
        "tokens_chunk_len": _configs.diffusion.tokens_chunk_len,
        "tokens_num_q": _configs.diffusion.tokens_num_q,
        "embedding_pretrained_model": _configs.diffusion.embedding_pretrained_model,
        "embedding_ckpt_url": _configs.diffusion.embedding_ckpt_url,
        "embedding_ckpt_sha256": _configs.diffusion.embedding_ckpt_sha256
    }
)

trainer_args.validation_batch_size = _configs.diffusion.batch_size
trainer_args.shuffle = _configs.diffusion.shuffle
trainer_args.random_seed = _configs.diffusion.random_seed
trainer_args.use_mixed_precision = True
trainer_args.use_grad_clip = _configs.diffusion.use_grad_clip
trainer_args.grad_clip_norm = _configs.diffusion.grad_clip_norm
trainer_args.grad_clip_value = _configs.diffusion.grad_clip_value
trainer_args.phases = [
    TrainingPhaseArgs(
        name="warmup",
        optimizer=_configs.diffusion.phases.warmup.optimizer,
        optimizer_kwargs=_configs.diffusion.phases.warmup.optimizer_args,
        learning_rate=_configs.diffusion.phases.warmup.learning_rate,
        max_epochs=_configs.diffusion.phases.warmup.epochs,
        batch_size=_configs.diffusion.batch_size,
        max_batches=_configs.diffusion.phases.warmup.max_batches,
        ckpt_interval=_configs.diffusion.ckpt_interval
    ),
    TrainingPhaseArgs(
        name="warmup2",
        optimizer=_configs.diffusion.phases.warmup2.optimizer,
        optimizer_kwargs=_configs.diffusion.phases.warmup2.optimizer_args,
        learning_rate=_configs.diffusion.phases.warmup2.learning_rate,
        max_epochs=_configs.diffusion.phases.warmup2.epochs,
        batch_size=_configs.diffusion.batch_size,
        max_batches=_configs.diffusion.phases.warmup2.max_batches,
        ckpt_interval=_configs.diffusion.ckpt_interval,
        max_loss=_configs.diffusion.phases.warmup2.max_loss
    ),
    TrainingPhaseArgs(
        name="main",
        optimizer=_configs.diffusion.phases.main.optimizer,
        optimizer_kwargs=_configs.diffusion.phases.main.optimizer_args,
        learning_rate=_configs.diffusion.phases.main.learning_rate,
        max_epochs=_configs.diffusion.phases.main.epochs,
        batch_size=_configs.diffusion.batch_size,
        ckpt_interval=_configs.diffusion.ckpt_interval
    )
]

if __name__ == "__main__":
    if not torch.cuda.is_available():
        trainer_args.parallel_enabled = False
    trainer_class_name = "ama_prof_divi.training.trainers.acoustic.DiffusionTrainer"
    start_trainer(trainer_class_name, trainer_args)