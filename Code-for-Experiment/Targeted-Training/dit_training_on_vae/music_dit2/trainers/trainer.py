import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import deepspeed
from typing import Any, Dict, Tuple
from pathlib import Path
from collections import OrderedDict
import get_accelerator
from deepspeed.monitor.monitor import MonitorMaster
import deepspeed
from deepspeed.runtime.config import DeepSpeedConfig
import get_hparams
import Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.dist as dist
from .dataset import DiTDataset, NUM_LOCAL_RANKS

class MusicDiTTrainer:
    def __init__(self, model_engine: deepspeed.DeepSpeedEngine, lr_scheduler: Any):
        self.model_engine = model_engine
        self.lr_scheduler = lr_scheduler
        self.rank = dist.get_rank()
        self.local_rank = dist.get_local_rank()
        self.world_size = dist.get_world_size()
        self.device = (torch.device(get_accelerator().device_name(), self.local_rank) if (self.local_rank > -1) and get_accelerator().is_available() else torch.device("cpu"))
        self.logger = deepspeed.logger
        root_path = Path(__file__).parent.parent.parent
        hparams = get_hparams(root_path)
        self.config = hparams.training.trainer
        ds_config = DeepSpeedConfig(model_engine.config)
        self.monitor = MonitorMaster(monitor_config=ds_config.monitor_config)
        self.mini_batch_size = self.model_engine.train_micro_batch_size_per_gpu()
        self.total_batch_size = self.mini_batch_size * self.world_size
        self.training_ds, self.validation_ds = DiTDataset.make_datasets()
        if self.rank % NUM_LOCAL_RANKS == 0:
            self.logger.info(f"For rank {self.rank}, Training dataset: {len(self.training_ds)} items"
                             f"Validation dataset: {len(self.validation_ds)} items.")
        trimmed_training_ds_len = self.trim_datasets_(self.training_ds)
        trimmed_validation_ds_len = self.trim_datasets_(self.validation_ds)
        self.max_vae_length = self.config.max_vae_length
        self.max_clap_length = self.max_vae_length // hparams.music_dit.vae_to_clap_ratio
        self.max_lyrics_length = self.config.max_lyrics_length
        if self.rank == 0:
            self.logger.info(f"Training datasets are trimmed to {trimmed_training_ds_len}, "
                             f"validation datasets are trimmed to {trimmed_validation_ds_len}.")
        self.training_sampler = DistributedSampler(self.training_ds,
                                                   num_replicas=NUM_LOCAL_RANKS,
                                                   shuffle=True,
                                                   seed=self.config.random_seed,
                                                   rank=self.local_rank)
        self.training_dataloader = DataLoader(self.training_ds,
                                              batch_size=self.mini_batch_size,
                                              sampler=self.training_sampler,
                                              num_workers=self.config.dataloader_num_workers,
                                              collate_fn=self.training_ds.collate_fn)
        self.validation_sampler = DistributedSampler(self.validation_ds,
                                                     num_replicas=NUM_LOCAL_RANKS,
                                                     shuffle=True,
                                                     seed=self.config.random_seed,
                                                     rank=self.local_rank)
        self.validation_dataloader = DataLoader(self.validation_ds,
                                                batch_size=self.mini_batch_size,
                                                sampler=self.validation_sampler,
                                                num_workers=self.config.dataloader_num_workers,
                                                collate_fn=self.validation_ds.collate_fn)
        self.total_num_steps = len(self.training_dataloader) * self.config.num_epochs
        self.lr_scheduler.total_num_steps = self.total_num_steps
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

    def trim_datasets_(self, dataset: DiTDataset) -> int:
        dist.barrier()
        ds_lens = torch.zeros(dist.get_world_size(), device=self.device)
        ds_lens[self.rank] = len(dataset)
        for i in range(dist.get_world_size()):
            t = ds_lens[i]
            dist.broadcast(t, src=i)
            ds_lens[i] = t
        batch_size_per_node = self.mini_batch_size * NUM_LOCAL_RANKS
        min_ds_len = int(ds_lens.min() / batch_size_per_node) * batch_size_per_node
        dataset.trim_to_length_(min_ds_len)
        dist.barrier()
        return min_ds_len

    def load_checkpoint(self):
        if self.rank == 0:
            self.logger.info(f"Trying to load checkpoint from {self.config.checkpoint_dir}...")
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(exist_ok=True, parents=False)
        _, client_state = self.model_engine.load_checkpoint(load_dir=str(ckpt_dir),
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

    def save_checkpoint(self, epoch: int, step: int, total_steps: int):
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(exist_ok=True, parents=False)
        client_state = {
            "current_epoch": epoch,
            "current_step": step,
            "current_total_steps": total_steps
        }
        self.model_engine.save_checkpoint(save_dir=str(ckpt_dir),
                                          client_state=client_state)
        if self.rank == 0:
            self.logger.info("Checkpoints saved. Epoch = %d, step = %d", epoch, step)

    def _get_data_from_batch(self, batch, dtype=torch.bfloat16) -> Dict[str, torch.Tensor]:
        return {
            "vae": batch["vae"]["data"].to(self.device).to(dtype).transpose(1, -1)[:, :self.max_vae_length, :],
            "vae_mask": batch["vae"]["padding_mask"].to(self.device).to(dtype)[:, :self.max_vae_length],
            "clap": batch["clap"]["data"].to(self.device).to(dtype).transpose(1, -1)[:, :self.max_clap_length, :],
            "clap_mask": batch["clap"]["padding_mask"].to(self.device).to(dtype)[:, :self.max_clap_length],
            "lyrics": batch["lyrics"]["data"].to(self.device).to(dtype).squeeze(1)[:, :self.max_lyrics_length] if "lyrics" in batch else None,
            "lyrics_mask": batch["lyrics"]["padding_mask"].to(self.device).to(dtype)[:, :self.max_lyrics_length] if "lyrics" in batch else None
        }

    def validate_loop(self, training_loss_dict: OrderedDict[str, torch.Tensor], current_step: int):
        for key in training_loss_dict:
            dist.all_reduce(training_loss_dict[key], op=dist.ReduceOp.SUM)
            training_loss_dict[key] /= self.world_size
        if self.rank == 0:
            self.logger.info(f"Validation at step {current_step}...")
        self.model_engine.eval()
        step = 0
        loss_dict = OrderedDict({
            "val_mse_loss": torch.Tensor([0.0]).to(self.device),
            "val_vb_loss": torch.Tensor([0.0]).to(self.device),
            "val_total_loss": torch.Tensor([0.0]).to(self.device),
            "val_inference_loss": torch.Tensor([0.0]).to(self.device)
        })
        with torch.no_grad():
            for step, batch in enumerate(self.validation_dataloader):
                data = self._get_data_from_batch(batch)
                loss = self.model_engine(data["vae"],
                                         data["vae_mask"],
                                         data["clap"],
                                         data["clap_mask"],
                                         data["lyrics"],
                                         data["lyrics_mask"],
                                         self.rank,
                                         self.world_size)
                loss_dict["val_mse_loss"] += loss["mse_loss"]
                loss_dict["val_vb_loss"] += loss["vb_loss"]
                loss_dict["val_total_loss"] += loss["total_loss"]
                inference_loss = self.model_engine.test_inference(
                    vae=data["vae"],
                    vae_mask=data["vae_mask"],
                    clap=data["clap"],
                    clap_mask=data["clap_mask"],
                    lyrics=data["lyrics"],
                    lyrics_mask=data["lyrics_mask"],
                    rank=self.rank,
                    world_size=self.world_size)
                loss_dict["val_inference_loss"] += inference_loss
                if step > self.config.num_validation_steps:
                    break
        scale = (step + 1) * self.world_size
        for key in loss_dict:
            dist.all_reduce(loss_dict[key], op=dist.ReduceOp.SUM)
            loss_dict[key] /= scale
        loss_dict = OrderedDict(**training_loss_dict, **loss_dict)
        if self.rank == 0:
            self.logger.info(f"Validation at step {current_step}/{self.total_num_steps}: "
                             f"mse loss = {loss_dict['val_mse_loss'].item()}, "
                             f"vb loss = {loss_dict['val_vb_loss'].item()}, "
                             f"total loss = {loss_dict['val_total_loss'].item()}, "
                             f"inference loss = {loss_dict['val_inference_loss'].item()}")
        global_steps = self.model_engine.global_samples // self.total_batch_size
        event_list = [
            ("Validation/" + key, value, global_steps)
            for key, value in loss_dict.items()
        ]
        self.monitor.write_events(event_list)

    def train_step(self, batch: Dict, current_step: int, epoch: int, step_in_epoch: int, epoch_length: int) -> Tuple[OrderedDict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        self.model_engine.train()
        self.model_engine.zero_grad()
        data = self._get_data_from_batch(batch)
        if random.random() <= self.config.lyrics_dropout:
            data["lyrics"] = None
            data["lyrics_mask"] = None
            lyrics_dropout = True
        else:
            lyrics_dropout = False
        if self.config.log_lyrics_length:
            if data["lyrics"] is not None:
                self.logger.info("Rank %d, step %d, lyrics: %s", self.rank, current_step, data["lyrics"].shape)
            else:
                self.logger.info("Rank %d, step %d, lyrics: %s", self.rank, current_step, "Dropped" if lyrics_dropout else "None")
        loss_dict = self.model_engine(data["vae"],
                                      data["vae_mask"],
                                      data["clap"],
                                      data["clap_mask"],
                                      data["lyrics"],
                                      data["lyrics_mask"],
                                      self.rank,
                                      self.world_size)
        assert loss_dict["total_loss"].dim() == 0
        self.model_engine.backward(loss_dict["total_loss"])
        self.model_engine.step()
        if self.rank == 0:
            if "x_start_min" in loss_dict:
                self.logger.info("x_start: min=%f, max=%f, mean=%f, std=%f",
                                 loss_dict["x_start_min"].item(),
                                 loss_dict["x_start_max"].item(),
                                 loss_dict["x_start_mean"].item(),
                                 loss_dict["x_start_std"].item())
            if "noise_min" in loss_dict:
                self.logger.info("noise: min=%f, max=%f, mean=%f, std=%f",
                                 loss_dict["noise_min"].item(),
                                 loss_dict["noise_max"].item(),
                                 loss_dict["noise_mean"].item(),
                                 loss_dict["noise_std"].item())
            if "predicted_noise_min" in loss_dict:
                self.logger.info("predicted noise: min=%f, max=%f, mean=%f, std=%f",
                                 loss_dict["predicted_noise_min"].item(),
                                 loss_dict["predicted_noise_max"].item(),
                                 loss_dict["predicted_noise_mean"].item(),
                                 loss_dict["predicted_noise_std"].item())
            if "predicted_log_variance_min" in loss_dict:
                self.logger.info("predicted log variance: min=%f, max=%f, mean=%f, std=%f",
                                 loss_dict["predicted_log_variance_min"].item(),
                                 loss_dict["predicted_log_variance_max"].item(),
                                 loss_dict["predicted_log_variance_mean"].item(),
                                 loss_dict["predicted_log_variance_std"].item())
            self.logger.info(f"Training step {current_step}/{self.total_num_steps} "
                             f"(epoch {epoch}, step {step_in_epoch}/{epoch_length}): "
                             f"mse_loss = {loss_dict['mse_loss'].item()}, "
                             f"vb_loss = {loss_dict['vb_loss'].item()}, "
                             f"total_loss = {loss_dict['total_loss'].item()}")
        global_steps = self.model_engine.global_samples // self.total_batch_size
        self.monitor.write_events([
            ("Train/mse_loss", loss_dict["mse_loss"], global_steps),
            ("Train/vb_loss", loss_dict["vb_loss"], global_steps),
            ("Train/total_loss", loss_dict["total_loss"], global_steps)
        ])
        loss_dict = OrderedDict({
            "trn_mse_loss": loss_dict["mse_loss"],
            "trn_vb_loss": loss_dict["vb_loss"],
            "trn_loss": loss_dict["total_loss"]
        })
        metrics_dict = {}
        return loss_dict, metrics_dict

    def start_training(self):
        if self.rank == 0:
            self.logger.info(f"Start training from epoch {self.initial_epoch}...")
        total_steps = self.initial_epoch * len(self.training_dataloader)
        trn_loss_dict = OrderedDict()
        trn_metrics_dict = {}
        for epoch in range(self.initial_epoch, self.config.num_epochs):
            self.training_sampler.set_epoch(epoch)
            self.validation_sampler.set_epoch(epoch)
            epoch_length = len(self.training_dataloader)
            for step, batch in enumerate(self.training_dataloader):
                total_steps += 1
                if self.initial_step > 0:
                    if self.rank == 0:
                        self.logger.info(f"Skipping step {step}...")
                    if step <= self.initial_step:
                        continue
                    self.initial_step = 0
                trn_loss_dict, trn_metrics_dict = self.train_step(
                    batch=batch,
                    current_step=total_steps,
                    epoch=epoch,
                    step_in_epoch=step,
                    epoch_length=epoch_length)
                if total_steps % self.config.validation_interval == 0:
                    self.validate_loop(training_loss_dict=trn_loss_dict,
                                       current_step=total_steps)
                    deepspeed.comm.log_summary()
                if self.config.checkpoint_interval >= 0 and total_steps % self.config.checkpoint_interval == 0:
                    for key in trn_metrics_dict.keys():
                        dist.all_reduce(trn_metrics_dict[key], op=dist.ReduceOp.SUM)
                        trn_metrics_dict[key] /= self.world_size
                    self.save_checkpoint(epoch=epoch,
                                         step=step,
                                         total_steps=total_steps)
                if self.config.cuda_cache_cleanup_interval >= 0 and \
                   total_steps % self.config.cuda_cache_cleanup_interval == 0:
                    torch.cuda.empty_cache()
            if self.rank == 0:
                self.logger.info(f"Epoch {epoch} completed.")
            self.validate_loop(training_loss_dict=trn_loss_dict,
                               current_step=total_steps)
            for key in trn_metrics_dict.keys():
                dist.all_reduce(trn_metrics_dict[key], op=dist.ReduceOp.SUM)
                trn_metrics_dict[key] /= self.world_size
            self.save_checkpoint(epoch=epoch + 1,
                                 step=0,
                                 total_steps=total_steps)
            deepspeed.comm.log_summary()
        dist.barrier()
        if self.rank == 0:
            self.logger.info("Done.")