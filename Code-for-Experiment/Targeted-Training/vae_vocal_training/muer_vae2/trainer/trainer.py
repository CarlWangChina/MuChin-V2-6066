import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Any, Dict, Tuple
from collections import OrderedDict
from pathlib import Path
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.monitor.monitor import MonitorMaster
from deepspeed.runtime.config import DeepSpeedConfig
from ama_prof_divi_common.utils import get_hparams
import ama_prof_divi_common.utils.dist_wrapper as dist
from Code_for_Experiment.Targeted_Training.vae_vocal_training.ama-prof-divi_vae2.trainer.dataset import AudioDatasetOnline
from .losses import *

NUM_LOCAL_RANKS = 8
datas = [
    'datas/audio-original-mp3',
    'datas/gansheng',
]

def check_nan_loss(loss, name, logger):
    if torch.isnan(loss).any():
        logger.info(f"Loss {name} has NaN values.")
        raise ValueError(f"Loss {name} has NaN values.")
    return loss

def check_for_nans(model):
    for name, param in model.named_parameters():
        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
            print(f"NaN or Inf in gradients of {name}")
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"NaN or Inf in parameters of {name}")
    raise ValueError("NaN or Inf found in model parameters or gradients")

class VAE2Trainer:
    def __init__(self,
                 model_engine_g: deepspeed.DeepSpeedEngine,
                 lr_scheduler_g: Any,
                 model_engine_d: deepspeed.DeepSpeedEngine,
                 lr_scheduler_d: Any):
        self.model_engine_g = model_engine_g
        self.lr_scheduler_g = lr_scheduler_g
        self.model_engine_d = model_engine_d
        self.lr_scheduler_d = lr_scheduler_d
        self.rank = dist.get_rank()
        self.local_rank = dist.get_local_rank()
        self.world_size = dist.get_world_size()
        self.device = (torch.device(get_accelerator().device_name(), self.local_rank)
                       if (self.local_rank > -1) and get_accelerator().is_available()
                       else torch.device("cpu"))
        self.datatype = next(self.model_engine_g.parameters()).dtype
        self.logger = deepspeed.logger
        hparams = get_hparams()
        self.config = hparams.training.trainer
        ds_config = DeepSpeedConfig(model_engine_g.config)
        self.monitor = MonitorMaster(monitor_config=ds_config.monitor_config)
        self.mini_batch_size = self.model_engine_g.train_micro_batch_size_per_gpu()
        self.total_batch_size = self.mini_batch_size * self.world_size
        self.training_ds, self.validation_ds = AudioDatasetOnline.make_datasets(
            chunk_size=hparams.training.dataset.chunk_size, data_pths=datas, sample_rate=hparams.vae.sampling_rate
        )
        self.logger.info(f"For rank {self.rank}, Training dataset: {len(self.training_ds)} items; "
                         f"Validation dataset: {len(self.validation_ds)} items.")
        self.training_sampler = DistributedSampler(self.training_ds,
                                                   num_replicas=NUM_LOCAL_RANKS,
                                                   shuffle=True,
                                                   seed=self.config.random_seed,
                                                   rank=self.local_rank)
        self.training_dataloader = DataLoader(self.training_ds,
                                              batch_size=self.mini_batch_size,
                                              sampler=self.training_sampler,
                                              num_workers=self.config.dataloader_num_workers,
                                              )
        self.validation_sampler = DistributedSampler(self.validation_ds,
                                                     num_replicas=NUM_LOCAL_RANKS,
                                                     shuffle=True,
                                                     seed=self.config.random_seed,
                                                     rank=self.local_rank)
        self.validation_dataloader = DataLoader(self.validation_ds,
                                                batch_size=self.mini_batch_size,
                                                sampler=self.validation_sampler,
                                                num_workers=self.config.dataloader_num_workers,
                                                )
        self.total_num_steps = len(self.training_dataloader) * self.config.num_epochs
        self.lr_scheduler_g.total_num_steps = self.total_num_steps
        self.lr_scheduler_d.total_num_steps = self.total_num_steps
        self.discriminator_warmup_steps = self.config.discriminator_warmup_steps
        self.kl_annealing_steps = self.config.kl_annealing_steps
        self.loss_weights = hparams.training.loss_weights
        self.l1_loss = get_l1_loss().to(self.device)
        self.l2_loss = get_l2_loss().to(self.device)
        self.kld_loss = get_kl_loss().to(self.device)
        self.mel_loss = get_mel_loss().to(self.device)
        self.stft_loss = get_stft_loss().to(self.device)
        self.sisnr_loss = get_sisnr_loss().to(self.device)

        self.initial_epoch = 0
        self.initial_step = 0
        self.load_checkpoint()

    def close(self):
        self.training_ds.close()
        self.validation_ds.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def load_checkpoint(self):
        if self.rank == 0:
            self.logger.info(f"Trying to load checkpoint from {self.config.checkpoint_dir}...")
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(exist_ok=True, parents=True)
        ckpt_dir_g = ckpt_dir / "g"
        ckpt_dir_d = ckpt_dir / "d"
        ckpt_dir_g.mkdir(exist_ok=True, parents=True)
        ckpt_dir_d.mkdir(exist_ok=True, parents=True)
        _, client_state = self.model_engine_g.load_checkpoint(load_dir=str(ckpt_dir_g),
                                                              load_module_strict=False,
                                                              load_optimizer_states=False,
                                                              load_lr_scheduler_states=False)
        self.model_engine_d.load_checkpoint(load_dir=str(ckpt_dir_d),
                                            load_module_strict=False,
                                            load_optimizer_states=False,
                                            load_lr_scheduler_states=False)
        if client_state is not None:
            self.initial_epoch = client_state["current_epoch"]
            self.initial_step = client_state["current_step"]
            if self.rank == 0:
                self.logger.info("Checkpoint loaded. Epoch = %d, step = %d",
                                 self.initial_epoch,
                                 self.initial_step)
        dist.barrier()

    def save_checkpoint(self,
                        epoch: int,
                        step: int,
                        total_steps: int,
                        latent_mean_mean: float,
                        latent_mean_std: float,
                        latent_std_mean: float,
                        latent_std_std: float):
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(exist_ok=True, parents=True)
        ckpt_dir_g = ckpt_dir / "g"
        ckpt_dir_d = ckpt_dir / "d"
        ckpt_dir_g.mkdir(exist_ok=True, parents=True)
        ckpt_dir_d.mkdir(exist_ok=True, parents=True)
        client_state = {
            "current_epoch": epoch,
            "current_step": step,
            "current_total_steps": total_steps,
            "latent_mean_mean": latent_mean_mean,
            "latent_mean_std": latent_mean_std,
            "latent_std_mean": latent_std_mean,
            "latent_std_std": latent_std_std
        }
        self.model_engine_g.save_checkpoint(save_dir=str(ckpt_dir_g),
                                            client_state=client_state)
        self.model_engine_d.save_checkpoint(save_dir=str(ckpt_dir_d),
                                            client_state=client_state)
        if self.rank == 0:
            self.logger.info("Checkpoints saved. Epoch = %d, step = %d", epoch, step)

    def _get_loss_weight_kld(self, current_step: int):
        return self.loss_weights.kld_annealing * current_step / self.kl_annealing_steps \
            if current_step < self.kl_annealing_steps else self.loss_weights.kld

    def validate_loop(self,
                      training_loss_dict: OrderedDict[str, torch.Tensor],
                      current_step: int):
        for key in training_loss_dict:
            dist.all_reduce(training_loss_dict[key], op=dist.ReduceOp.SUM)
            training_loss_dict[key] /= self.world_size
        if self.rank == 0:
            self.logger.info(f"Validation at step {current_step}...")
        self.model_engine_g.eval()
        self.model_engine_d.eval()
        step = 0
        loss_dict = OrderedDict({
            "val_loss_l1": torch.Tensor([0.0]).to(self.device),
            "val_loss_l2": torch.Tensor([0.0]).to(self.device),
            "val_loss_mel": torch.Tensor([0.0]).to(self.device),
            "val_loss_kld": torch.Tensor([0.0]).to(self.device),
            "val_loss_stft": torch.Tensor([0.0]).to(self.device),
            "val_loss_sisnr": torch.Tensor([0.0]).to(self.device),
            "val_loss_adv": torch.Tensor([0.0]).to(self.device),
            "val_loss_gen": torch.Tensor([0.0]).to(self.device),
            "val_loss_fm": torch.Tensor([0.0]).to(self.device),
            "val_loss_gen_total": torch.Tensor([0.0]).to(self.device),
            "val_loss_total": torch.Tensor([0.0]).to(self.device)
        })
        with torch.no_grad():
            for step, batch in enumerate(self.validation_dataloader):
                wav, mask = batch
                audio = wav.to(self.device).to(self.datatype)
                audio_recon, latent_mean, latent_log_var = self.model_engine_g(audio,
                                                                               rank=self.rank,
                                                                               world_size=self.world_size)
                loss_adv, loss_gen, loss_fm = self.model_engine_d(audio, audio_recon.detach())
                loss_adv *= self.loss_weights.adv
                loss_gen *= self.loss_weights.gen
                loss_fm *= self.loss_weights.fm
                loss_l1 = self.l1_loss(audio, audio_recon) * self.loss_weights.l1
                loss_l2 = self.l2_loss(audio, audio_recon) * self.loss_weights.l2
                loss_kld = torch.Tensor([0.0]).to(self.device)
                loss_mel = self.mel_loss(audio, audio_recon) * self.loss_weights.mel
                loss_stft = self.stft_loss(audio, audio_recon) * self.loss_weights.stft
                loss_sisnr = self.sisnr_loss(audio, audio_recon) * self.loss_weights.sisnr
                loss_gen_total = loss_l1 + loss_l2 + loss_mel + loss_kld + loss_stft + loss_sisnr + loss_gen + loss_fm
                loss_total = loss_adv + loss_gen_total
                loss_dict["val_loss_l1"] += loss_l1
                loss_dict["val_loss_l2"] += loss_l2
                loss_dict["val_loss_mel"] += loss_mel
                loss_dict["val_loss_kld"] += loss_kld
                loss_dict["val_loss_stft"] += loss_stft
                loss_dict["val_loss_sisnr"] += loss_sisnr
                loss_dict["val_loss_adv"] += loss_adv
                loss_dict["val_loss_gen"] += loss_gen
                loss_dict["val_loss_fm"] += loss_fm
                loss_dict["val_loss_gen_total"] += loss_gen_total
                loss_dict["val_loss_total"] += loss_total
                if step > self.config.num_validation_steps:
                    break
        scale = (step + 1) * self.world_size
        for key in loss_dict:
            dist.all_reduce(loss_dict[key], op=dist.ReduceOp.SUM)
            loss_dict[key] /= scale
        loss_dict = OrderedDict(**training_loss_dict, **loss_dict)
        if self.rank == 0:
            self.logger.info(f"Validation at step {current_step}/{self.total_num_steps}: "
                             f"loss_l1 = {loss_dict['val_loss_l1'].item()}, "
                             f"loss_l2 = {loss_dict['val_loss_l2'].item()}, "
                             f"loss_mel = {loss_dict['val_loss_mel'].item()},"
                             f"loss_kld = {loss_dict['val_loss_kld'].item()},"
                             f"loss_adv = {loss_dict['val_loss_adv'].item()},"
                             f"loss_total = {loss_dict['val_loss_total'].item()}")
        global_steps = self.model_engine_g.global_samples // self.total_batch_size
        event_list = [
            ("Validation/" + key, value, global_steps)
            for key, value in loss_dict.items()
        ]
        self.monitor.write_events(event_list)

    def train_step(self,
                   batch,
                   current_step: int,
                   epoch: int,
                   step_in_epoch: int,
                   epoch_length: int,
                   last_loss_mel: float) -> Tuple[OrderedDict[str, torch.Tensor], Dict[str, torch.Tensor], float]:
        self.model_engine_g.train()
        self.model_engine_d.train()
        self.model_engine_g.zero_grad()
        self.model_engine_d.zero_grad()
        audio = batch.to(self.device).to(self.datatype)
        assert audio.dim() == 3, f"Invalid audio dim = {audio.dim()}"
        audio_recon, latent_mean, latent_log_var = self.model_engine_g(audio,
                                                                       rank=self.rank,
                                                                       world_size=self.world_size)
        if current_step >= self.discriminator_warmup_steps >= 0:
            loss_adv, _, _ = self.model_engine_d(audio, audio_recon.detach())
            loss_adv *= self.loss_weights.adv
            self.model_engine_d.backward(loss_adv)
            self.model_engine_d.step()
        else:
            loss_adv = torch.Tensor([0.0]).to(self.device)
        loss_l1 = self.l1_loss(audio, audio_recon) * self.loss_weights.l1
        loss_l2 = self.l2_loss(audio, audio_recon) * self.loss_weights.l2
        loss_kld = torch.Tensor([0.0]).to(self.device)
        loss_mel = self.mel_loss(audio, audio_recon) * self.loss_weights.mel
        loss_stft = self.stft_loss(audio, audio_recon) * self.loss_weights.stft
        loss_sisnr = self.sisnr_loss(audio, audio_recon) * self.loss_weights.sisnr
        loss_l1 = check_nan_loss(loss_l1, "loss_l1", self.logger)
        loss_l2 = check_nan_loss(loss_l2, "loss_l2", self.logger)
        loss_kld = check_nan_loss(loss_kld, "loss_kld", self.logger)
        loss_mel = check_nan_loss(loss_mel, "loss_mel", self.logger)
        loss_stft = check_nan_loss(loss_stft, "loss_stft", self.logger)
        loss_sisnr = check_nan_loss(loss_sisnr, "loss_sisnr", self.logger)
        loss_adv = check_nan_loss(loss_adv, "loss_adv", self.logger)
        if loss_mel.item() > last_loss_mel * 3:
            for i in range(len(batch)):
                loss_mel_i = self.mel_loss(audio[i:i+1, ...], audio_recon[i:i+1, ...]) * self.loss_weights.mel
                if loss_mel_i.item() > last_loss_mel * 3:
                    self.logger.warning("Possible outliers detected: song_id: %s, loss_mel = %f, last_loss_mel = %f",
                                        batch[i],
                                        loss_mel_i.item(), last_loss_mel)
        _, loss_gen, loss_fm = self.model_engine_d(audio, audio_recon)
        loss_gen *= self.loss_weights.gen
        loss_fm *= self.loss_weights.fm
        loss_gen_total = loss_l1 + loss_l2 + loss_mel + loss_kld + loss_stft + loss_sisnr + loss_gen + loss_fm
        loss_total = loss_adv + loss_gen_total
        self.model_engine_g.backward(loss_gen_total)
        self.model_engine_g.step()
        if self.rank == 0:
            self.logger.info("latent_mean: %s, min=%f, max=%f, mean=%f, std=%f",
                             latent_mean.shape,
                             latent_mean.min().item(),
                             latent_mean.max().item(),
                             latent_mean.mean().item(),
                             latent_mean.std().item())
            self.logger.info("latent_log_var: %s, min=%f, max=%f, mean=%f, std=%f",
                             latent_log_var.shape,
                             latent_log_var.min().item(),
                             latent_log_var.max().item(),
                             latent_log_var.mean().item(),
                             latent_log_var.std().item())
            self.logger.info("audio: %s, min=%f, max=%f, mean=%f, std=%f",
                             audio.shape,
                             audio.min().item(),
                             audio.max().item(),
                             audio.mean().item(),
                             audio.std().item())
            self.logger.info("audio_recon: %s, min=%f, max=%f, mean=%f, std=%f",
                             audio_recon.shape,
                             audio_recon.min().item(),
                             audio_recon.max().item(),
                             audio_recon.mean().item(),
                             audio_recon.std().item())
            self.logger.info(f"Training step {current_step}/{self.total_num_steps} "
                             f"(epoch {epoch}, step {step_in_epoch}/{epoch_length}): "
                             f"loss_l1 = {loss_l1.item()},"
                             f"loss_l2 = {loss_l2.item()},"
                             f"loss_mel = {loss_mel.item()},"
                             f"loss_kld = {loss_kld.item()},"
                             f"loss_stft = {loss_stft.item()},"
                             f"loss_sisnr = {loss_sisnr.item()},"
                             f"loss_adv = {loss_adv.item()},"
                             f"loss_gen = {loss_gen.item()},"
                             f"loss_fm = {loss_fm.item()},"
                             f"loss_gen_total = {loss_gen_total.item()},"
                             f"loss_total = {loss_total.item()}")
        global_steps = self.model_engine_g.global_samples // self.total_batch_size
        self.monitor.write_events([
            ("Train/loss_l1", loss_l1, global_steps),
            ("Train/loss_l2", loss_l2, global_steps),
            ("Train/loss_mel", loss_mel, global_steps),
            ("Train/loss_kld", loss_kld, global_steps),
            ("Train/loss_stft", loss_stft, global_steps),
            ("Train/loss_sisnr", loss_sisnr, global_steps),
            ("Train/loss_adv", loss_adv, global_steps),
            ("Train/loss_gen", loss_gen, global_steps),
            ("Train/loss_fm", loss_fm, global_steps),
            ("Train/loss_gen_total", loss_gen_total, global_steps),
            ("Train/loss_total", loss_total, global_steps),
            ("Train/mean_latent_mean", latent_mean.mean(), global_steps),
            ("Train/mean_latent_std", latent_mean.std(), global_steps),
            ("Train/mean_logvar_mean", latent_log_var.mean(), global_steps),
        ])
        if dist.get_rank() == 0 and current_step % self.config.checkpoint_interval == 0:
            self.monitor.tb_monitor.summary_writer.add_audio("Train/audio", audio[-1], global_steps, sample_rate=16000)
            self.monitor.tb_monitor.summary_writer.add_audio("Train/audio_recon", audio_recon[-1], global_steps, sample_rate=16000)
        loss_dict = OrderedDict({
            "trn_loss_l1": loss_l1,
            "trn_loss_l2": loss_l2,
            "trn_loss_mel": loss_mel,
            "trn_loss_kld": loss_kld,
            "trn_loss_stft": loss_stft,
            "trn_loss_sisnr": loss_sisnr,
            "trn_loss_adv": loss_adv,
            "trn_loss_gen": loss_gen,
            "trn_loss_fm": loss_fm,
            "trn_loss_gen_total": loss_gen_total,
            "trn_loss_total": loss_total
        })
        metrics_dict = {
            "latent_mean_mean": latent_mean.mean(),
            "latent_mean_std": latent_mean.std(),
            "latent_std_mean": latent_log_var.exp().mean().sqrt(),
            "latent_std_std": latent_log_var.exp().std().sqrt()
        }
        return loss_dict, metrics_dict, loss_mel.item()

    def start_training(self):
        if self.rank == 0:
            self.logger.info(f"Start training from epoch {self.initial_epoch} ...")
        total_steps = self.initial_epoch * len(self.training_dataloader)
        trn_loss_dict = OrderedDict()
        trn_metrics_dict = {
            "latent_mean_mean": torch.Tensor([0.0]).to(self.device),
            "latent_mean_std": torch.Tensor([0.0]).to(self.device),
            "latent_std_mean": torch.Tensor([0.0]).to(self.device),
            "latent_std_std": torch.Tensor([0.0]).to(self.device)
        }
        for epoch in range(self.initial_epoch, self.config.num_epochs):
            self.training_sampler.set_epoch(epoch)
            self.validation_sampler.set_epoch(epoch)
            epoch_length = len(self.training_dataloader)
            last_loss_mel = 100.0
            for step, batch in enumerate(self.training_dataloader):
                wav, mask = batch
                total_steps += 1
                trn_loss_dict, trn_metrics_dict, last_loss_mel = self.train_step(
                    batch=wav,
                    current_step=total_steps,
                    epoch=epoch,
                    step_in_epoch=step,
                    epoch_length=epoch_length,
                    last_loss_mel=last_loss_mel)
                if total_steps % self.config.validation_interval == 0:
                    self.validate_loop(training_loss_dict=trn_loss_dict,
                                       current_step=total_steps)
                    deepspeed.comm.log_summary()
                if total_steps % self.config.checkpoint_interval == 0:
                    for key in trn_metrics_dict.keys():
                        dist.all_reduce(trn_metrics_dict[key], op=dist.ReduceOp.SUM)
                        trn_metrics_dict[key] /= self.world_size
                    self.save_checkpoint(epoch=epoch,
                                         step=step,
                                         total_steps=total_steps,
                                         latent_mean_mean=trn_metrics_dict["latent_mean_mean"].item(),
                                         latent_mean_std=trn_metrics_dict["latent_mean_std"].item(),
                                         latent_std_mean=trn_metrics_dict["latent_std_mean"].item(),
                                         latent_std_std=trn_metrics_dict["latent_std_std"].item())
            self.logger.info(f"Epoch {epoch} completed.")
            self.validate_loop(training_loss_dict=trn_loss_dict,
                               current_step=total_steps)
            for key in trn_metrics_dict.keys():
                dist.all_reduce(trn_metrics_dict[key], op=dist.ReduceOp.SUM)
                trn_metrics_dict[key] /= self.world_size
            self.save_checkpoint(epoch=epoch + 1,
                                 step=0,
                                 total_steps=total_steps,
                                 latent_mean_mean=trn_metrics_dict["latent_mean_mean"].item(),
                                 latent_mean_std=trn_metrics_dict["latent_mean_std"].item(),
                                 latent_std_mean=trn_metrics_dict["latent_std_mean"].item(),
                                 latent_std_std=trn_metrics_dict["latent_std_std"].item())
            deepspeed.comm.log_summary()
        dist.barrier()
        if self.rank == 0:
            self.logger.info("Done.")