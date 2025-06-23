import re
import os
import sys
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import logging
from torch.cuda.amp import GradScaler, autocast
from typing import Optional
from pathlib import Path
from tqdm.auto import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from ama_prof_divi.configs import get_hparams
from .trainer_args import TrainerArgs, TrainingPhaseArgs
from .training_logger import TrainingLogger
import MetricsCollector
logger = logging.getLogger(__name__)

class BaseTrainer(ABC):
    def __init__(self, args: TrainerArgs, *, device: torch.device = torch.device("cpu"), parallel_enabled: bool = False, world_size: int = 1, rank: int = 0, local_rank: int = 0, is_master: bool = True):
        super(BaseTrainer, self).__init__()
        self.args = args
        self.parallel_enabled = self.args.parallel_enabled
        self.phases = self.args.phases
        self.stop_training = False
        self.parallel_enabled = parallel_enabled
        self.rank = rank
        self.local_rank = local_rank
        self.is_master = is_master
        self.world_size = world_size
        self.hparams = get_hparams()
        self.device = device
        self.training_loss_list = []
        self.validation_loss_list = []

        assert len(self.phases) > 0, "No training phase is specified."
        self.checkpoint_dir = self._make_valid_dir_path(self.args.checkpoint_dir)
        self.log_dir = self._make_valid_dir_path(self.args.log_dir)
        self.data_dir = self._make_valid_dir_path(self.args.data_dir, read_only=True)
        if local_rank == 0:
            logger.info("Creating the trainer on device: " + str(self.device))
            logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
            logger.info(f"Log directory: {self.log_dir}")
            logger.info(f"Data directory: {self.data_dir}")
            logger.info("Loading datasets ...")
        self.training_dataset, self.validation_dataset = self.load_datasets(self.hparams, self.args, self.device)
        assert self.training_dataset is not None, "Training dataset is None."
        assert self.validation_dataset is not None, "Validation dataset is None."
        if local_rank == 0:
            logger.info(f"Training dataset: {len(self.training_dataset)} samples.")
            logger.info(f"Validation dataset: {len(self.validation_dataset)} samples.")
            logger.info("Creating data loaders ...")
        self.model = None
        self.model_has_no_parameters = False
        self.optimizer = None
        self.training_logger = None
        self.grad_scaler = GradScaler() if self.args.use_mixed_precision else None
        self.training_sampler = DistributedSampler(self.training_dataset, num_replicas=self.world_size, shuffle=self.args.shuffle, seed=self.args.random_seed, rank=self.rank) if self.parallel_enabled else None
        self.validation_sampler = DistributedSampler(self.validation_dataset, num_replicas=self.world_size, shuffle=self.args.shuffle, seed=self.args.random_seed, rank=self.rank) if self.parallel_enabled else None
        self.training_dataloaders = [
            DataLoader(self.training_dataset, batch_size=phase.batch_size, sampler=self.training_sampler, shuffle=None if self.training_sampler is not None else self.args.shuffle, num_workers=self.args.num_workers, collate_fn=self.collate_fn, persistent_workers=True) for phase in self.phases
        ]
        self.validation_dataloader = DataLoader(self.validation_dataset, batch_size=self.args.validation_batch_size, sampler=self.validation_sampler, shuffle=None if self.validation_sampler is not None else self.args.shuffle, num_workers=self.args.num_workers, collate_fn=self.collate_fn, persistent_workers=True)
        if local_rank == 0:
            logger.info("Syncing with other nodes.  If the program halts here, check the NCCL network connection.")
        if self.parallel_enabled:
            dist.barrier()
        if local_rank == 0:
            logger.info("Synced.")

    def _make_valid_dir_path(self, direct: str, read_only: bool = False) -> Path:
        dir_path = Path(direct)
        if not dir_path.exists():
            if self.args.mkdir_if_not_exist and not read_only:
                dir_path.mkdir(parents=True, exist_ok=True)
            else:
                raise FileNotFoundError(f"Directory {dir_path} does not exist.")
        else:
            assert dir_path.is_dir(), f"{dir_path} is not a directory."
        if not dir_path.is_absolute():
            dir_path = dir_path.resolve()
        return dir_path

    @abstractmethod
    def load_datasets(self, hparams: dict, trainer_args: TrainerArgs, device: torch.device) -> (Dataset, Dataset):
        ...

    @staticmethod
    def collate_fn(batch):
        return torch.stack(batch)

    @abstractmethod
    def get_model(self, hparams: dict, trainer_args: TrainerArgs, device: torch.device) -> nn.Module:
        ...

    @abstractmethod
    def get_criterion(self, hparams: dict, trainer_args: TrainerArgs, device: torch.device) -> Optional[nn.Module]:
        ...

    def _load_checkpoint(self, path: Path, current_phase: int) -> dict:
        states = torch.load(path, map_location="cpu")
        state_dict = states["state_dict"]
        if self.parallel_enabled:
            adjusted_state_dict = {}
            for key in state_dict.keys():
                adjusted_key = "module." + key if not key.startswith("module.") else key
                adjusted_state_dict[adjusted_key] = state_dict[key]
            state_dict = adjusted_state_dict
        self.model.load_state_dict(state_dict)
        if self.optimizer is not None:
            if "optimizer" in states and states["optimizer"] == self.phases[current_phase].optimizer:
                if self.local_rank == 0:
                    logger.info(f"Optimizer: {states['optimizer']}")
                self.optimizer.load_state_dict(states["optimizer_state_dict"])
        return {
            "current_phase": current_phase,
            "current_epoch": states["epoch"] - 1,
            "current_batch": states["batch"],
            "current_step": states["current_step"],
            "running_loss": states["running_loss"],
        }

    def _load_most_recent_checkpoint(self) -> dict:
        pattern = "^" + self.args.ckpt_file_prefix + r"_(\w+)_" + r"epoch_(\d+)_batch_(\d+)" + self.args.ckpt_file_postfix + "$"
        pattern = pattern.replace(".", r"\.")
        pattern = re.compile(pattern)
        phases_available = [phase.name for phase in self.phases]
        ckpt_list = []
        for file in os.listdir(self.checkpoint_dir):
            match = pattern.match(file)
            if match:
                path = Path(self.checkpoint_dir) / file
                phase = match.group(1)
                epoch = int(match.group(2)) - 1
                batch = int(match.group(3)) - 1
                if phase in phases_available:
                    ckpt_list.append((path, phase, epoch, batch))
        if len(ckpt_list) == 0:
            if self.local_rank == 0:
                logger.info(f"No checkpoint is found in {self.checkpoint_dir}.")
            return {
                "current_phase": 0,
                "current_epoch": 0,
                "current_batch": 0,
                "current_step": 0,
                "running_loss": 0.0,
            }
        ckpt_list.sort(key=lambda x: (phases_available.index(x[1]), x[2], x[3]))
        if self.local_rank == 0:
            logger.info(f"Loading the most recent checkpoint from {ckpt_list[-1][0]}...")
        current_phase = phases_available.index(ckpt_list[-1][1])
        return self._load_checkpoint(ckpt_list[-1][0], current_phase)

    def _save_checkpoint(self, phase: int, epoch: int, batch: int, current_step: int, running_loss: float, validation_loss: float):
        ckpt_file = (self.args.ckpt_file_prefix + f"_{self.phases[phase].name}_" + f"epoch_{epoch}_batch_{batch}" + self.args.ckpt_file_postfix)
        ckpt_path = self.checkpoint_dir / ckpt_file
        if self.local_rank == 0:
            logger.info(f"Saving checkpoint to {ckpt_file}...")
        state_dict = self.model.state_dict()
        if self.parallel_enabled:
            for key in state_dict.keys():
                assert key.startswith("module."), "The model is not in the DDP mode."
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        if self.optimizer is not None:
            states = {
                "phase": self.phases[phase].name,
                "epoch": epoch,
                "batch": batch,
                "current_step": current_step,
                "running_loss": running_loss,
                "validation_loss": validation_loss,
                "state_dict": state_dict,
                "optimizer": self.phases[phase].optimizer,
                "optimizer_state_dict": self.optimizer.state_dict()
            }
        else:
            states = {
                "phase": self.phases[phase].name,
                "epoch": epoch,
                "batch": batch,
                "current_step": current_step,
                "running_loss": running_loss,
                "state_dict": self.model.state_dict()
            }
        torch.save(states, ckpt_path)

    @abstractmethod
    def train_step(self, data: any, phase: str, epoch: int, batch: int, model: nn.Module, criterion: nn.Module) -> dict or torch.Tensor:
        ...

    @abstractmethod
    @torch.inference_mode()
    def validation_step(self, data: any, phase: str, epoch: int, batch: int, model: nn.Module, criterion: nn.Module) -> dict or torch.Tensor:
        ...

    def should_terminate_training(self, phase: TrainingPhaseArgs, epoch: int, batch: int, training_loss_list: [float], validation_loss_list: [float]) -> bool:
        return False

    def check_anomaly(self, *, phase_name: str, epoch: int, batch: int, training_result: Optional[any] = None, loss: Optional[torch.Tensor] = None) -> bool:
        if self.args.check_model_anomalies:
            for k, v in self.model.state_dict().items():
                if torch.any(torch.isnan(v)):
                    logger.error(f"In epoch {epoch}, batch {batch}, rank {self.rank}: Model state {k} contains NaN value.")
                    return True
        if loss is not None and torch.isnan(loss):
            logger.error(f"In epoch {epoch}, batch {batch}, rank {self.rank}: Loss is NaN.")
            return True
        return False

    def train_loop(self, phase: int, epoch: int, *, training_states: dict):
        if self.parallel_enabled:
            self.training_sampler.set_epoch(epoch)
        loader = self.training_dataloaders[phase]
        ckpt_interval = self.args.phases[phase].ckpt_interval
        max_batches = self.args.phases[phase].max_batches
        criterion = self.get_criterion(hparams=self.hparams, trainer_args=self.args, device=self.device)
        if self.local_rank == 0:
            logger.info(f"Training phase '{self.phases[phase].name}' epoch {epoch + 1}...")
        pbar = tqdm(total=len(loader), desc=f"Training '{self.phases[phase].name}' epoch {epoch + 1}") if self.local_rank == 0 else None
        running_batch = training_states["current_batch"]
        training_metrics = MetricsCollector(parallel_enabled=self.parallel_enabled, local_rank=self.local_rank, world_size=self.world_size, device=self.device)
        current_metrics = MetricsCollector(parallel_enabled=self.parallel_enabled, local_rank=self.local_rank, world_size=self.world_size, device=self.device)
        running_loss = training_states["running_loss"] * training_states["current_batch"]
        training_metrics["loss"] = torch.tensor([running_loss], device=self.device)
        loader_len = len(loader)
        loader_iter = iter(loader)
        for batch in range(min(training_states["current_batch"], loader_len)):
            _ = next(loader_iter)
            if self.local_rank == 0:
                pbar.update(1)
                pbar.set_postfix({"skipping": 1})
        for batch in range(training_states["current_batch"], loader_len + 100):
            try:
                data = next(loader_iter)
            except StopIteration:
                break
            if max_batches is not None and batch >= max_batches:
                break
            self.model.train()
            if self.optimizer is not None:
                self.optimizer.zero_grad(set_to_none=True)
            if self.args.use_mixed_precision:
                with autocast(dtype=torch.float16):
                    train_result = self.train_step(data, self.phases[phase].name, epoch, batch, self.model, criterion)
            else:
                train_result = self.train_step(data, self.phases[phase].name, epoch, batch, self.model, criterion)
            if torch.is_tensor(train_result):
                loss = train_result.to(self.device)
                training_metrics += {"loss": loss}
                current_metrics.metrics = {"current_loss": loss}
            else:
                assert "loss" in train_result, "The loss is not in the training returned dict."
                loss = train_result["loss"].to(self.device)
                training_metrics += train_result
                current_metrics.metrics = {"current_loss": train_result["loss"]}
                assert loss is not None, "The loss is None."
            if self.check_anomaly(phase_name=self.phases[phase].name, epoch=epoch + 1, batch=batch + 1, training_result=train_result, loss=loss):
                logger.fatal(f"In rank {self.rank}, anomaly found after forward propagation. Aborting.")
                sys.exit(1)
            if self.local_rank == 0:
                pbar.update(1)
                pbar.set_postfix({"loss": loss.item()})
            running_batch += 1
            training_states["current_step"] += 1
            if self.local_rank == 0:
                if self.training_logger is None:
                    log_dir = self.log_dir / self.args.phases[phase].name
                    log_dir.mkdir(exist_ok=True)
                    self.training_logger = TrainingLogger(log_dir, args=self.args, local_rank=self.local_rank)
            else:
                self.training_logger = None
            if self.training_logger is not None:
                self.training_logger.log_metrics(current_metrics, label="training", step=training_states["current_step"])
            if self.args.use_mixed_precision:
                self.grad_scaler.scale(loss).backward()
                if self.optimizer is not None:
                    self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                loss.backward()
                if self.optimizer is not None:
                    self.optimizer.step()
            if self.args.use_grad_clip:
                if self.args.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip_norm)
                elif self.args.grad_clip_value is not None:
                    nn.utils.clip_grad_value_(self.model.parameters(), self.args.grad_clip_value)
            if self.check_anomaly(phase_name=self.phases[phase].name, epoch=epoch + 1, batch=batch + 1):
                logger.fatal(f"In rank {self.rank}, anomaly found after backward propagation. Aborting.")
                sys.exit(1)
            if (batch + 1) % ckpt_interval == 0 or (batch + 1) == len(loader):
                training_metrics.all_divide_(running_batch)
                training_metrics.all_reduce_()
                validation_metrics = MetricsCollector(parallel_enabled=self.parallel_enabled, local_rank=self.local_rank, world_size=self.world_size, device=self.device)
                if self.local_rank == 0:
                    pbar.close()
                if self.optimizer is not None:
                    self.optimizer.zero_grad(set_to_none=True)
                self.validation_loop(phase, epoch, batch, metrics=validation_metrics, criterion=criterion)
                if self.local_rank == 0:
                    pbar = tqdm(total=len(loader), desc=f"Training '{self.phases[phase].name}' epoch {epoch + 1}")
                    pbar.update(batch)
                assert "loss" in validation_metrics, "The loss is not in the validation returned dict."
                self.training_loss_list.append(training_metrics["loss"].item())
                self.validation_loss_list.append(validation_metrics["loss"].item())
                if self.local_rank == 0:
                    logger.info(f"Epoch {epoch + 1} batch {batch + 1} running loss: {training_metrics['loss'].item()} validation loss: {validation_metrics['loss'].item()}")
                    self._save_checkpoint(phase, epoch + 1, batch + 1, training_states["current_step"], running_loss=training_metrics["loss"].item(), validation_loss=validation_metrics['loss'].item())
                    if self.training_logger is not None:
                        self.training_logger.log_metrics(training_metrics, label="training", step=training_states["current_step"])
                        self.training_logger.log_metrics(validation_metrics, label="validation", step=training_states["current_step"])
                if self.parallel_enabled:
                    dist.barrier()
                training_metrics.clear()
                running_batch = 0
                if self.should_terminate_training(self.phases[phase], epoch, batch, self.training_loss_list, self.validation_loss_list):
                    self.stop_training = True
                    break
        training_states["current_batch"] = 0
        if self.local_rank == 0:
            pbar.close()
        logger.info(f"Training phase '{self.phases[phase].name}', epoch {epoch + 1} finished.")
        if self.parallel_enabled:
            dist.barrier()

    def validation_loop(self, phase: int, epoch: int, batch: int, metrics: MetricsCollector, criterion: nn.Module):
        self.model.eval()
        num_validation_samples = min(len(self.validation_dataloader), self.args.num_validation_samples)
        if self.local_rank == 0:
            logger.info("Validating ...")
        pbar = tqdm(total=num_validation_samples, desc=f"Validating '{self.phases[phase].name}' epoch {epoch + 1} batch {batch + 1}", leave=True) if self.local_rank == 0 else None
        if self.parallel_enabled:
            self.validation_sampler.set_epoch(epoch)
        data_iter = iter(self.validation_dataloader)
        with torch.no_grad():
            for i in range(num_validation_samples):
                try:
                    data = next(data_iter)
                except StopIteration:
                    break
                if self.args.use_mixed_precision:
                    with autocast(dtype=torch.float16):
                        validation_result = self.validation_step(data, self.phases[phase].name, epoch, i, self.model, criterion)
                else:
                    validation_result = self.validation_step(data, self.phases[phase].name, epoch, i, self.model, criterion)
                if torch.is_tensor(validation_result):
                    loss = validation_result.to(self.device)
                    metrics += {"loss": loss}
                else:
                    assert "loss" in validation_result, "The loss is not in the validation returned dict."
                    loss = validation_result["loss"].to(self.device)
                    metrics += validation_result
                if self.local_rank == 0:
                    pbar.update(1)
                    pbar.set_postfix({"loss": loss.item()})
        metrics.all_divide_(num_validation_samples)
        metrics.all_reduce_()
        if self.local_rank == 0:
            pbar.close()

    def train(self):
        if self.model is None:
            if self.local_rank == 0:
                logger.info("Loading the model ...")
            model = self.get_model(self.hparams, self.args, self.device)
            if self.parallel_enabled and len(list(model.parameters())) > 0:
                self.model = DDP(model.to(self.device), device_ids=[self.local_rank]) if self.parallel_enabled else model.to(self.device)
                self.model_has_no_parameters = False
            else:
                self.model = model.to(self.device)
                self.model_has_no_parameters = (len(list(model.parameters())) == 0)
        if self.local_rank == 0:
            logger.info("Model and optimizers are loaded.")
        self.training_loss_list = []
        self.validation_loss_list = []
        self.stop_training = False
        if self.args.use_pretrain_model:
            training_states = self._load_most_recent_checkpoint()
        else:
            training_states = {
                "current_phase": 0,
                "current_epoch": 0,
                "current_batch": 0,
                "current_step": 0,
                "running_loss": 0.0,
            }
        if self.parallel_enabled:
            dist.barrier()
        for phase in range(training_states["current_phase"], len(self.phases)):
            phase_args = self.phases[phase]
            if self.local_rank == 0:
                logger.info(f"Start training [{phase_args.name}]...")
            if self.optimizer is not None:
                del self.optimizer
            self.optimizer = None
            if not self.model_has_no_parameters and phase_args.optimizer is not None:
                optimizer_cls = getattr(torch.optim, phase_args.optimizer)
                assert optimizer_cls is not None, f"Optimizer {phase_args.optimizer} is not supported."
                if phase_args.optimizer_kwargs is not None:
                    self.optimizer = optimizer_cls(self.model.parameters(), lr=phase_args.learning_rate, **phase_args.optimizer_kwargs)
                else:
                    self.optimizer = optimizer_cls(self.model.parameters(), lr=phase_args.learning_rate)
            if self.optimizer is not None and phase_args.lr_scheduler is not None:
                lr_scheduler = getattr(torch.optim.lr_scheduler, phase_args.lr_scheduler)
                assert lr_scheduler is not None, f"LR scheduler {phase_args.lr_scheduler} is not supported."
                if phase_args.lr_scheduler_kwargs is not None:
                    lr_scheduler = lr_scheduler(self.optimizer, **phase_args.lr_scheduler_kwargs)
                else:
                    lr_scheduler = lr_scheduler(self.optimizer)
            else:
                lr_scheduler = None
            if lr_scheduler is not None:
                if self.local_rank == 0 and training_states["current_epoch"] > 0:
                    logger.info(f"Fast forwarding the LR scheduler for phase {phase}...")
                for epoch in range(training_states["current_epoch"]):
                    lr_scheduler.step()
            self.training_logger = None
            for epoch in range(training_states["current_epoch"], self.phases[phase].max_epochs):
                self.train_loop(phase, epoch, training_states=training_states)
                if lr_scheduler is not None:
                    lr_scheduler.step()
                if self.stop_training:
                    break
            if self.stop_training:
                break
            training_states["current_epoch"] = 0
            training_states["current_step"] = 0
            training_states["current_batch"] = 0