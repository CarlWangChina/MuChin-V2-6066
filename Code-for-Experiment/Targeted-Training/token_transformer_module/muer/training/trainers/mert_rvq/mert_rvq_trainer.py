import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional
from ama_prof_divi.utils.logging import get_logger
from ama_prof_divi.training import BaseTrainer, TrainerArgs
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.modules.rvq.rvq import ResidualVectorQuantization
from ...datasets import MertDataset

logger = get_logger(__name__)

def default(d, key: str, default_value):
    return d[key] if key in d else default_value

class MertRVQTrainer(BaseTrainer):
    def __init__(self, args: TrainerArgs, *, device: torch.device, parallel_enabled: bool, world_size: int, rank: int, local_rank: int, is_master: bool):
        super(MertRVQTrainer, self).__init__(args, device=device, parallel_enabled=parallel_enabled, world_size=world_size, rank=rank, local_rank=local_rank, is_master=is_master)

    def get_model(self, hparams: dict, trainer_args: TrainerArgs, device: torch.device) -> nn.Module:
        extra_args = trainer_args.extra_args
        assert extra_args is not None, "The extra arguments should not be None."
        model = ResidualVectorQuantization(dim=extra_args["feature_dim"],
                                           num_quantizers=extra_args["num_quantizers"],
                                           codebook_size=extra_args["codebook_size"],
                                           similarity=extra_args["similarity"],
                                           ema_update=False,
                                           learnable_codebook=True)
        return model.to(device)

    def load_datasets(self, hparams: dict, trainer_args: TrainerArgs, device: torch.device) -> (Dataset, Dataset):
        extra_args = trainer_args.extra_args
        assert extra_args is not None, "The extra arguments should not be None."
        window_size = extra_args["window_size"]
        feature_dim = default(extra_args, "feature_dim", 1024)
        feature_rate = default(extra_args, "feature_rate", 75)
        stride = default(extra_args, "stride", -1)
        content = "vectors"
        last_window = "drop"
        cache_file = self.checkpoint_dir / "cache.pt"
        return MertDataset.load_datasets(data_dir=self.data_dir,
                                         mert_file_pattern=trainer_args.data_file_pattern,
                                         cache_file=cache_file,
                                         window_size=window_size,
                                         feature_dim=feature_dim,
                                         feature_rate=feature_rate,
                                         stride=stride,
                                         content=content,
                                         last_window=last_window,
                                         use_data_files_proportion=trainer_args.use_data_files_proportion,
                                         training_dataset_proportion=trainer_args.training_dataset_proportion,
                                         local_rank=self.local_rank)

    @staticmethod
    def collate_fn(batch):
        return MertDataset.collate_fn(batch)

    def get_criterion(self, hparams: dict, trainer_args: TrainerArgs, device: torch.device) -> Optional[nn.Module]:
        return None

    def train_step(self, data: any, phase: str, epoch: int, batch: int, model: nn.Module, criterion: nn.Module) -> torch.Tensor:
        x = data.to(self.device)
        y = model(x)
        loss = y["all_losses"].sum()
        return loss

    def validation_step(self, data: any, phase: str, epoch: int, batch: int, model: nn.Module, criterion: nn.Module) -> torch.Tensor:
        x = data.to(self.device)
        y = model(x)
        loss = F.mse_loss(x, y["quantized_out"])
        return loss