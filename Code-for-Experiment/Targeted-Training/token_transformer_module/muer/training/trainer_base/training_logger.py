from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import TrainerArgs
import MetricsCollector

class TrainingLogger:
    def __init__(self, log_path: Path, *, args: TrainerArgs, local_rank: int = 0):
        self.log_path = log_path
        self.args = args
        self.local_rank = local_rank
        assert self.log_path.exists() and self.log_path.is_dir(), \
            f"The log directory {self.log_path} does not exist."
        self.writer = SummaryWriter(log_dir=str(self.log_path))

    def log_scalar(self, tag: str, scalar_value: float, *, step: int):
        if self.local_rank == 0:
            self.writer.add_scalar(tag, scalar_value, global_step=step, new_style=True)

    def log_hparams(self, hparams_dict: dict, metric_dict: dict):
        if self.local_rank == 0:
            self.writer.add_hparams(hparam_dict=hparams_dict, metric_dict=metric_dict)

    def log_metrics(self, metrics: MetricsCollector, *, label: str, step: int):
        if self.local_rank == 0:
            for key, value in metrics.items():
                if key.startswith("_"):
                    continue
                if key != "":
                    key = f"{key[0].upper()}{key[1:]}"
                self.writer.add_scalar(f"{key}/{label}", value, global_step=step, new_style=True)