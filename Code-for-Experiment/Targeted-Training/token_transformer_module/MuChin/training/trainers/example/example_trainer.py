import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional
from ama_prof_divi.utils import safe_softmax
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger
from ama_prof_divi.training import BaseTrainer, TrainerArgs

logger = get_logger(__name__)

class ExampleDataset(Dataset):
    def __init__(self, training: bool = True):
        super(ExampleDataset, self).__init__()
        self.training = training
        self.num_items = 300 if training else 10
        self.data = torch.randint(0, 100, (self.num_items,), dtype=torch.long)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_items

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(100, 16)
        self.ln1 = nn.Linear(16, 128)
        self.ln2 = nn.Linear(128, 128)
        self.norm = nn.LayerNorm(128)

    def forward(self, x):
        x = self.embedding(x)
        x = F.relu(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.ln2(x)
        x = self.norm(x)
        x = safe_softmax(x, dim=-1)
        return x

class ExampleTrainer(BaseTrainer):
    def __init__(self, args: TrainerArgs, *, device: torch.device, parallel_enabled: bool, world_size: int, rank: int, local_rank: int, is_master: bool):
        super(ExampleTrainer, self).__init__(args, device=device, parallel_enabled=parallel_enabled, world_size=world_size, rank=rank, local_rank=local_rank, is_master=is_master)
        if local_rank == 0:
            logger.info("Example trainer created.")

    def load_datasets(self, hparams: dict, trainer_args: TrainerArgs, device: torch.device) -> (Dataset, Dataset):
        return ExampleDataset(training=True), ExampleDataset(training=False)

    def get_model(self, hparams: dict, trainer_args: TrainerArgs, device: torch.device) -> nn.Module:
        return Net().to(device)

    def get_criterion(self, hparams: dict, trainer_args: TrainerArgs, device: torch.device) -> Optional[nn.Module]:
        return nn.CrossEntropyLoss().to(device)

    def _run_model(self, data, model, criterion) -> dict:
        x = data.to(self.device)
        x = model(x)
        loss = criterion(x, data.to(self.device))
        if self.device.type == "mps":
            x = x.to("cpu")
            data = data.to("cpu")
        else:
            data = data.to(self.device)
        y = torch.argmax(x, dim=-1)
        accuracy = (y == data).sum() / data.size(0)
        return {
            "loss": loss,
            "accuracy": accuracy
        }

    def train_step(self, data: any, phase: str, epoch: int, batch: int, model: nn.Module, criterion: nn.Module) -> dict:
        return self._run_model(data, model, criterion)

    def validation_step(self, data: any, phase: str, epoch: int, batch: int, model: nn.Module, criterion: nn.Module) -> dict:
        return self._run_model(data, model, criterion)