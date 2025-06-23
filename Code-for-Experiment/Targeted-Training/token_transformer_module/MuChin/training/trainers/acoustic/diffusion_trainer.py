import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Optional
from einops import rearrange
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.logging import get_logger
import Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.modules.diffusion.model as DiffusionModel
from ama_prof_divi.training import BaseTrainer, TrainerArgs, TrainingPhaseArgs

logger = get_logger(__name__)

class DiffusionTrainer(BaseTrainer):
    def __init__(self, args: TrainerArgs, *, device: torch.device, parallel_enabled: bool, world_size: int, rank: int, local_rank: int, is_master: bool):
        super(DiffusionTrainer, self).__init__(args, device=device, parallel_enabled=parallel_enabled, world_size=world_size, rank=rank, local_rank=local_rank, is_master=is_master)
        if local_rank == 0:
            logger.info(f"DiffusionTrainer created: {args}")

    def load_datasets(self, hparams: dict, trainer_args: TrainerArgs, device: torch.device) -> (Dataset, Dataset):
        extra_args = trainer_args.extra_args
        assert extra_args is not None, "The extra arguments should not be None."
        cache_file = self.checkpoint_dir / "cache.pt"
        return DiffusionDataset.load_datasets(data_dir=self.data_dir, data_file_pattern=trainer_args.data_file_pattern, window_size=extra_args["window_size"], cache_file=cache_file, use_data_files_proportion=trainer_args.use_data_files_proportion, training_dataset_proportion=trainer_args.training_dataset_proportion, clap_dim=extra_args["clap_dim"], tokens_chunk_len=extra_args["tokens_chunk_len"], tokens_num_q=extra_args["tokens_num_q"], local_rank=self.local_rank)

    def get_model(self, hparams: dict, trainer_args: TrainerArgs, device: torch.device) -> nn.Module:
        return DiffusionModel(hparams=hparams, extra_args=trainer_args.extra_args, local_rank=self.local_rank, parallel_enabled=trainer_args.parallel_enabled, device=device)

    def get_criterion(self, hparams: dict, trainer_args: TrainerArgs, device: torch.device) -> Optional[nn.Module]:
        return None

    def train_step(self, data: any, phase: str, epoch: int, batch: int, model: nn.Module, criterion: nn.Module) -> dict:
        tokens = data["tokens"].to(self.device)
        claps = rearrange(data["claps"], 'b d -> b 1 d').to(self.device)
        result = self.model(acoustic_tokens=tokens, prompt_embedding=claps)
        return {
            "_tokens": tokens,
            "_claps": claps,
            "_result": result,
            "loss": result["loss"]
        }

    def validation_step(self, data: any, phase: str, epoch: int, batch: int, model: nn.Module, criterion: nn.Module) -> dict:
        tokens = data["tokens"].to(self.device)
        claps = rearrange(data["claps"], 'b d -> b 1 d').to(self.device)
        return {
            "loss": self.model(acoustic_tokens=tokens, prompt_embedding=claps)["loss"]
        }

    @staticmethod
    def collate_fn(batch):
        return DiffusionDataset.collate_fn(batch)

    def check_anomaly(self, *, phase_name: str, epoch: int, batch: int, training_result: Optional[any] = None, loss: Optional[torch.Tensor] = None) -> bool:
        anomaly = super(DiffusionTrainer, self).check_anomaly(phase_name=phase_name, epoch=epoch, batch=batch, training_result=training_result, loss=loss)
        if training_result is not None:
            tokens = training_result["_tokens"]
            claps = training_result["_claps"]
            result = training_result["_result"]
            if self.args.check_model_anomalies:
                if torch.isnan(tokens).any() or torch.isnan(claps).any() or torch.isnan(loss).any():
                    anomaly = True
                for k, v in result.items():
                    if torch.isnan(v).any():
                        anomaly = True
                        break
            else:
                tokens = None
                claps = None
                result = None
        if anomaly:
            dump_error_file = self.checkpoint_dir / f"dump_error_{phase_name}_{epoch}_{batch}_rank{self.rank}.pt"
            model_state_dict = self.model.state_dict()
            if self.parallel_enabled:
                for key in model_state_dict.keys():
                    assert key.startswith("module."), "The model is not in the DDP mode."
                model_state_dict = {k[len("module."):]: v for k, v in model_state_dict.items()}
            if training_result is not None:
                torch.save({
                    "phase": phase_name,
                    "epoch": epoch,
                    "batch": batch,
                    "rank": self.rank,
                    "tokens": tokens,
                    "claps": claps,
                    "loss": result["loss"],
                    "latent": result["latent"],
                    "time_steps": result["time_steps"],
                    "noise": result["noise"],
                    "noisy_latent": result["noisy_latent"],
                    "context": result["context"],
                    "noise_pred": result["noise_pred"],
                    "model": model_state_dict,
                }, dump_error_file)
            else:
                torch.save({
                    "phase": phase_name,
                    "epoch": epoch,
                    "batch": batch,
                    "rank": self.rank,
                    "model": model_state_dict,
                }, dump_error_file)
            logger.warning(f"Dumped the error data to {dump_error_file}.")
            return True
        return False