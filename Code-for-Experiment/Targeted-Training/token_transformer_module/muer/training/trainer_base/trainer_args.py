from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class TrainingPhaseArgs:
    name: str = "main"
    optimizer: str = "Adam"
    learning_rate: float = 1e-3
    optimizer_kwargs: Optional[dict] = None
    lr_scheduler: Optional[str] = None
    lr_scheduler_kwargs: Optional[dict] = None
    max_epochs: int = 100
    batch_size: int = 1
    max_batches: Optional[int] = None
    ckpt_interval: int = 100
    max_loss: Optional[float] = None

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict):
        return TrainingPhaseArgs(**d)

@dataclass
class TrainerArgs:
    name: str
    data_dir: str
    checkpoint_dir: str
    log_dir: str
    data_file_pattern: str
    ckpt_file_prefix: Optional[str] = "checkpoint"
    ckpt_file_postfix: str = ".ckpt"
    log_file_prefix: Optional[str] = "log"
    log_file_postfix: str = ".log"
    random_seed: int = 0
    mkdir_if_not_exist: bool = True
    recursive: bool = True
    shuffle: bool = True
    use_data_files_proportion: float = 1.0
    training_dataset_proportion: float = 0.8
    num_workers: int = 2
    num_validation_samples: int = 100
    validation_batch_size: int = 2
    use_pretrain_model: bool = True
    use_mixed_precision: bool = False
    check_model_anomalies: bool = False
    use_grad_clip: bool = False
    grad_clip_norm: Optional[float] = None
    grad_clip_value: Optional[float] = None
    parallel_enabled: bool = True
    parallel_backend: str = "nccl"
    parallel_strategy: str = "ddp"
    master_addr: str = "localhost"
    master_port: int = 29500
    phases = [TrainingPhaseArgs()]
    extra_args: dict = None

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict):
        return TrainerArgs(**d)