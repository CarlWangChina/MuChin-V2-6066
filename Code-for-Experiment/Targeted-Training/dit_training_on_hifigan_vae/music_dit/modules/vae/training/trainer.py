import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, Dict
import music_dit.utils.dist_wrapper as dist
from music_dit.utils import get_logger, get_hparams
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.monitor.monitor import MonitorMaster
from deepspeed.runtime.config import DeepSpeedConfig
from .losses import (
    MultiScaleMelSpectrogramLoss,
    SISNRLoss,
    get_adversarial_loss
)
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.modules.vae.training.datasets.dataset import VAETrainingDataset

class VAETrainer:
    def __init__(self,
                 model_engine: deepspeed.DeepSpeedEngine,
                 optimizer: torch.optim.Optimizer,
                 training_dataset: VAETrainingDataset,
                 validation_dataset: VAETrainingDataset,
                 lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None):
        hparams = get_hparams()
        self.model_engine = model_engine
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.rank = dist.get_rank()
        self.local_rank = dist.get_local_rank()
        self.world_size = dist.get_world_size()
        self.device = (torch.device(get_accelerator().device_name(), self.local_rank)
                       if (self.local_rank > -1) and get_accelerator().is_available()
                       else torch.device("cpu"))
        self.logger = get_logger(__name__)
        self.l1_loss = nn.L1Loss().to(self.device)
        self.spec_loss = MultiScaleMelSpectrogramLoss(
            sampling_rate=hparams.vae.sampling_rate,
            range_start=hparams.vae.training.msspec.range_start,
            range_end=hparams.vae.training.msspec.range_end,
            n_mels=hparams.vae.training.msspec.n_mels,
            f_min=hparams.vae.training.msspec.f_min,
            f_max=hparams.vae.training.msspec.f_max,
            normalized=hparams.vae.training.msspec.normalized,
            alphas=hparams.vae.training.msspec.alphas,
            floor_level=hparams.vae.training.msspec.floor_level
        ).to(self.device)
        self.adv_loss = get_adversarial_loss(model=model_engine, device=self.device)
        self.sisnr_loss = SISNRLoss(sample_rate=hparams.vae.sampling_rate).to(self.device)
        self.loss_weights = {
            'adv': hparams.vae.training.loss_weights.adv,
            'feat': hparams.vae.training.loss_weights.feat,
            'l1': hparams.vae.training.loss_weights.l1,
            'msspec': hparams.vae.training.loss_weights.msspec,
            'sisnr': hparams.vae.training.loss_weights.sisnr
        }
        self.training_sampler = DistributedSampler(training_dataset,
                                                   num_replicas=self.world_size,
                                                   shuffle=False,
                                                   rank=self.rank)
        self.training_dataloader = DataLoader(training_dataset,
                                              batch_size=self.model_engine.train_micro_batch_size_per_gpu(),
                                              sampler=self.training_sampler,
                                              num_workers=hparams.training.vae.dataloader_num_workers,
                                              collate_fn=VAETrainingDataset.collate_fn)
        self.validation_sampler = DistributedSampler(validation_dataset,
                                                     num_replicas=self.world_size,
                                                     shuffle=True,
                                                     seed=hparams.training.vae.random_seed,
                                                     rank=self.rank)
        self.validation_dataloader = DataLoader(validation_dataset,
                                                batch_size=self.model_engine.train_micro_batch_size_per_gpu(),
                                                sampler=self.validation_sampler,
                                                num_workers=hparams.training.vae.dataloader_num_workers,
                                                collate_fn=VAETrainingDataset.collate_fn)
        ds_config = DeepSpeedConfig(model_engine.config)
        ds_config.monitor_config.tensorboard.output_path = hparams.training.vae.log_dir
        self.monitor = MonitorMaster(ds_config.monitor_config)
        self.num_epochs = hparams.training.vae.num_epochs
        if self.lr_scheduler is not None:
            self.lr_scheduler.total_num_steps = len(self.training_dataloader) * self.num_epochs
        self.checkpoint_dir = hparams.training.vae.checkpoint_dir
        self.num_validation_steps = hparams.training.vae.num_validation_steps
        self.warmup_steps_without_discriminator = hparams.training.vae.warmup_steps_without_discriminator
        self.initial_epoch = 0
        self.initial_step = 0

    def run_step(self,
                 data_batch: torch.Tensor,
                 current_step: int) -> torch.Tensor:
        assert torch.is_tensor(data_batch)
        self.model_engine.train()
        self.optimizer.zero_grad()
        x = data_batch.to(self.device)
        y_pred = self.model_engine(x)
        y = x.clone().type_as(y_pred)
        d_loss, feat_loss, fake_loss, real_loss = self.adv_loss(y_pred, y)
        losses = {
            "adv": d_loss if current_step > self.warmup_steps_without_discriminator
            else torch.Tensor([0.0]).sum().to(self.device),
            "feat": feat_loss,
            "l1": self.l1_loss(y_pred, y),
            "msspec": self.spec_loss(y_pred, y),
            "sisnr": self.sisnr_loss(y_pred, y)
        }
        loss = torch.stack([self.loss_weights[k] * v for k, v in losses.items()]).sum()
        self.model_engine.backward(loss)
        self.optimizer.step()
        if self.rank == 0:
            self.logger.info(f"total_loss = {loss},"
                             f"adv_loss = {d_loss} [fake={fake_loss}, real={real_loss}],"
                             f"feat_loss = {losses['feat']},"
                             f"l1_loss = {losses['l1']},"
                             f"msspec_loss = {losses['msspec']},"
                             f"snr_loss = {losses['sisnr']}")
        self.monitor.write_events([
            ("Train/adv_loss", losses["adv"], self.model_engine.global_samples),
            ("Train/adv_fake_loss", fake_loss, self.model_engine.global_samples),
            ("Train/adv_real_loss", real_loss, self.model_engine.global_samples),
            ("Train/feat_loss", losses["feat"], self.model_engine.global_samples),
            ("Train/l1_loss", losses["l1"], self.model_engine.global_samples),
            ("Train/msspec_loss", losses["msspec"], self.model_engine.global_samples),
            ("Train/snsnr_loss", losses['sisnr'], self.model_engine.global_samples),
            ("Train/total_loss", loss, self.model_engine.global_samples)
        ])
        return loss

    def train(self):
        if self.rank == 0:
            self.logger.info("Start training...")
        loss = torch.Tensor([0.0]).to(self.device)
        total_steps = 0
        for epoch in range(self.initial_epoch, self.num_epochs):
            self.training_sampler.set_epoch(epoch)
            batch_len = len(self.training_dataloader)
            for step, batch in enumerate(self.training_dataloader):
                total_steps += 1
                if self.initial_step > 0:
                    if self.rank == 0:
                        self.logger.info(f"Skipping step {step}...")
                    if step < self.initial_step:
                        continue
                    self.initial_step = 0
                loss = self.run_step(batch, total_steps)
                if self.rank == 0:
                    self.logger.info(f"Training Epoch: {epoch}, "
                                     f"Step: {step}/{batch_len}, "
                                     f"Loss: {loss.item()}")
            self.validate_and_log(epoch=epoch, step=total_steps, training_loss=loss)
            if self.rank == 0:
                self.logger.info(f"Saving checkpoint to {self.checkpoint_dir}...")
            self.model_engine.save_checkpoint(save_dir=self.checkpoint_dir,
                                              client_state={"current_epoch": epoch + 1,
                                                            "current_step": 0})

    def load_checkpoints(self):
        _, client_state = self.model_engine.load_checkpoint(load_dir=self.checkpoint_dir,
                                                            load_module_strict=True,
                                                            load_optimizer_states=True,
                                                            load_lr_scheduler_states=True)
        if client_state is not None:
            self.initial_epoch = client_state["current_epoch"]
            self.initial_step = client_state["current_step"]
            if self.rank == 0:
                self.logger.info("Checkpoint loaded. Epoch = %d, step = %d",
                                 self.initial_epoch,
                                 self.initial_step)
        dist.barrier()

    def validate_and_log(self, epoch: int, step: int, training_loss: torch.Tensor):
        if self.rank == 0:
            self.logger.info(f"Validating epoch {epoch + 1}, step {step + 1}...")
        dist.all_reduce(training_loss, op=dist.ReduceOp.SUM)
        training_loss /= self.world_size
        validation_result = self.validate()
        self.monitor.write_events([
            ("Validation/training_loss", training_loss, self.model_engine.global_samples),
            ("Validation/l1_loss", validation_result["l1_loss"], self.model_engine.global_samples),
            ("Validation/msspec_loss", validation_result["msspec_loss"], self.model_engine.global_samples),
            ("Validation/sisnr_loss", validation_result["sisnr_loss"], self.model_engine.global_samples)
        ])

    def validate(self) -> Dict[str, torch.Tensor]:
        self.model_engine.eval()
        total_l1_loss = torch.Tensor([0.0]).to(self.device)
        total_msspec_loss = torch.Tensor([0.0]).to(self.device)
        total_sisnr_loss = torch.Tensor([0.0]).to(self.device)
        with (torch.no_grad()):
            for step, batch in enumerate(self.validation_dataloader):
                x = batch.to(self.device)
                y_pred = self.model_engine(x)
                y = x.clone().type_as(y_pred)
                total_l1_loss += self.l1_loss(y_pred, y)
                total_msspec_loss += self.spec_loss(y_pred, y)
                total_sisnr_loss += self.sisnr_loss(y_pred, y)
                if step > self.num_validation_steps:
                    break
        dist.all_reduce(total_l1_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_msspec_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_sisnr_loss, op=dist.ReduceOp.SUM)
        scale = (step + 1) * self.world_size
        total_l1_loss /= scale
        total_msspec_loss /= scale
        total_sisnr_loss /= scale
        return {
            "l1_loss": total_l1_loss,
            "msspec_loss": total_msspec_loss,
            "sisnr_loss": total_sisnr_loss
        }