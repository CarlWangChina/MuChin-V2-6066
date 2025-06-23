import torch
import torch.nn as nn
from typing import Optional
from einops import rearrange
from ama_prof_divi.utils.logging import get_logger
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.training.trainer_base.trainer_args import TrainerArgs
import TrainerArgs
from .diffusion_trainer import DiffusionTrainer
import DiffusionTrainer
import EmbeddingModel
logger = get_logger(__name__)

class EmbeddingTrainer(DiffusionTrainer):
    def __init__(self, args: TrainerArgs, *, device: torch.device, parallel_enabled: bool, world_size: int, rank: int, local_rank: int, is_master: bool):
        super(EmbeddingTrainer, self).__init__(args, device=device, parallel_enabled=parallel_enabled, world_size=world_size, rank=rank, local_rank=local_rank, is_master=is_master)
        pad_id = args.extra_args["pad_id"]
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
        if local_rank == 0:
            logger.info(f"EmbeddingTrainer created: {args}")

    def get_model(self, hparams: dict, trainer_args: TrainerArgs, device: torch.device) -> nn.Module:
        return EmbeddingModel(vocab_size=trainer_args.extra_args["vocab_size"], dim=trainer_args.extra_args["dim"], local_rank=self.local_rank).to(device)

    def get_criterion(self, hparams: dict, trainer_args: TrainerArgs, device: torch.device) -> Optional[nn.Module]:
        return self.criterion

    def _forward(self, model: nn.Module, data: any, criterion: nn.Module) -> any:
        tokens = data["tokens"].to(self.device)
        logits = model(tokens)
        loss = criterion(rearrange(logits, 'b ... d -> b d ...'), tokens)
        prediction = torch.argmax(logits, dim=-1)
        accuracy = torch.Tensor([0.0]).to(self.device)
        for i in range(tokens.size(0)):
            accuracy += torch.all(prediction[i] == tokens[i]).float()
        accuracy /= tokens.size(0)
        return {
            "loss": loss,
            "accuracy": accuracy,
        }

    def train_step(self, data: any, phase: str, epoch: int, batch: int, model: nn.Module, criterion: nn.Module) -> dict:
        return self._forward(model, data, criterion)

    def validation_step(self, data: any, phase: str, epoch: int, batch: int, model: nn.Module, criterion: nn.Module) -> dict:
        return self._forward(model, data, criterion)