import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import deepspeed
import ama_prof_divi_common.utils.dist_wrapper as dist
import torch
from deepspeed.accelerator import get_accelerator
from deepspeed.monitor.monitor import MonitorMaster
from deepspeed.runtime.config import DeepSpeedConfig
from ama_prof_divi_codec import ama_prof_diviCodec
from ama_prof_divi_common.minio import MinioClient
from ama_prof_divi_common.threading import ThreadPool
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.hparams import get_hparams
from ray.data import DataIterator, Dataset
from .losses import *


class VAE2Trainer:
    def __init__(self,
                 dataset: Dataset,
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
        split_indices = [int(hparams.training.dataset.use_data_proportions *
                             hparams.training.dataset.train_proportion *
                             dataset.count()),
                         int(hparams.training.dataset.use_data_proportions *
                             dataset.count())]
        self.training_ds, self.validation_ds, _ = dataset.split_at_indices(split_indices)
        self.logger.info(f"For rank {self.rank}, Training dataset: {self.training_ds.count()} items; "
                         f"Validation dataset: {self.validation_ds.count()} items.")
        self.prefetch_factor = int(hparams.training.dataset.prefetch_factor)
        group_size = self.world_size * self.prefetch_factor
        num_groups = min((self.training_ds.count() // group_size // self.world_size) * self.world_size,
                         self.world_size)
        self.logger.info(f"For rank {self.rank}, repartitioning training set into {num_groups} groups...")
        self.training_ds = self.training_ds.repartition(num_blocks=num_groups)
        self.training_ds = self.training_ds.materialize()
        self.total_num_steps = self.training_ds.count() * self.config.num_epochs
        self.lr_scheduler_g.total_num_steps = self.total_num_steps
        self.lr_scheduler_d.total_num_steps = self.total_num_steps
        thread_pool = ThreadPool(num_workers=self.config.dataloader_num_workers)
        minio_client = MinioClient(thread_pool=thread_pool, part_size=hparams.training.dataset.minio_part_size)
        self.codec = ama_prof_diviCodec(thread_pool=thread_pool, minio_client=minio_client)
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
        self.skip_data = False
        self.num_channels = self.model_engine_g.num_channels
        self.chunk_length = self.model_engine_g.chunk_length
        self.audio_duration = (self.chunk_length * 1.2) / self.model_engine_g.sampling_rate
        self.load_checkpoint()
        dist.barrier()
        self.logger.info(f"Trainer initialized for rank {self.rank}.")

    def close(self):
        self.training_ds.close()
        self.validation_ds.close()

    def __enter__(self):
        pass

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
                                                              load_module_strict=True,
                                                              load_optimizer_states=True,
                                                              load_lr_scheduler_states=True)
        self.model_engine_d.load_checkpoint(load_dir=str(ckpt_dir_d),
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

    def _load_random_audio_chunks(self, files: List[str], durations: List[float]) -> torch.Tensor:
        datap = torch.rand(self.world_size, self.mini_batch_size)
        start_pos = [(durations[i] - self.audio_duration) * datap[self.rank, i] for i in range(len(files))]
        end_pos = [start_pos[i] + self.audio_duration for i in range(len(files))]
        result_data = torch.zeros(len(files), self.num_channels, self.chunk_length,
                                  device=self.device,
                                  dtype=self.datatype)
        try:
            _, audio_data = self.codec.decode_files(files,
                                                    out_channels=self.num_channels,
                                                    out_sampling_rate=self.model_engine_g.sampling_rate,
                                                    start_pos=start_pos,
                                                    end_pos=end_pos,
                                                    return_pt=True)
        except Exception as e:
            self.logger.error("On rank %d, Error decoding files: %s", self.rank, e)
            return result_data
        for i in range(len(files)):
            length = min(audio_data[i].size(-1), self.chunk_length)
            result_data[i, :, :length] = audio_data[i][:, :length]
            if length < self.chunk_length:
                self.logger.warning("On rank %d, audio length [%d] is less than chunk_length: %d < %d",
                                   self.rank, i, length, self.chunk_length)
        return result_data

    def _collate_fn(self, data: Any):
        obj_ids = [str(item) for item in data["obj_id"]]
        files = [str(item) for item in data["filename"]]
        if self.skip_data:
            return {
                "obj_id": obj_ids,
                "files": files,
                "durations": data["duration"],
                "audio": None
            }
        else:
            return {
                "obj_id": obj_ids,
                "files": files,
                "audio": self._load_random_audio_chunks(files, data["duration"])
            }

    def get_data_iterator(self,
                          dataset: Dataset,
                          *,
                          epoch: Optional[int] = None,
                          shuffle: bool = False) -> Tuple[DataIterator, int]:
        ds = dataset.randomize_block_order(seed=self.config.random_seed + epoch) if shuffle else dataset
        ds = ds.split(self.world_size, equal=True)[self.rank]
        ds = ds.materialize()
        data_iterator = ds.iter_torch_batches(
            batch_size=self.mini_batch_size,
            collate_fn=lambda x: self._collate_fn(x),
            local_shuffle_buffer_size=self.mini_batch_size * self.prefetch_factor if shuffle else None,
            prefetch_batches=self.prefetch_factor,
            local_shuffle_seed=self.config.random_seed + epoch if shuffle else None,
            drop_last=True
        )
        return data_iterator, ds.count() // self.mini_batch_size

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
        validation_data_iterator, validation_epoch_size = self.get_data_iterator(self.validation_ds)
        if self.rank == 0:
            self.logger.info("Validation epoch size: %d", validation_epoch_size)
        with torch.no_grad():
            for step, batch in enumerate(validation_data_iterator):
                if batch is None:
                    continue
                audio = batch["audio"]
                audio_recon, latent_mean, latent_log_var = self.model_engine_g(audio,
                                                                               rank=self.rank,
                                                                               world_size=self.world_size)
                loss_adv, loss_gen, loss_fm = self.model_engine_d(audio, audio_recon.detach())
                loss_adv *= self.loss_weights.adv
                loss_gen *= self.loss_weights.gen
                loss_fm *= self.loss_weights.fm
                loss_l1 = self.l1_loss(audio, audio_recon) * self.loss_weights.l1
                loss_l2 = self.l2_loss(audio, audio_recon) * self.loss_weights.l2
                loss_kld = self.kld_loss(latent_mean, latent_log_var) * self._get_loss_weight_kld(current_step)
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
                if step > self.config.num_validation_steps >= 0:
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
        self.model_engine_g.train()
        self.model_engine_d.train()

    def train_step(self,
                   batch: Dict,
                   current_step: int,
                   epoch: int,
                   step_in_epoch: int,
                   epoch_length: int,
                   last_loss_mel: float) -> Tuple[OrderedDict[str, torch.Tensor], Dict[str, torch.Tensor], float]:
        self.model_engine_g.zero_grad()
        self.model_engine_d.zero_grad()
        audio = batch["audio"]
        assert audio.dim() == 3, f"Invalid audio dim = {audio.dim()}"
        start_time = time.time()
        audio_recon, latent_mean, latent_log_var = self.model_engine_g(audio,
                                                                       rank=self.rank,
                                                                       world_size=self.world_size)
        current_time = time.time()
        forward_g_time = current_time - start_time
        start_time = time.time()
        if current_step >= self.discriminator_warmup_steps >= 0:
            loss_adv, _, _ = self.model_engine_d(audio, audio_recon.detach())
            loss_adv *= self.loss_weights.adv
            self.model_engine_d.backward(loss_adv)
            self.model_engine_d.step()
        else:
            loss_adv = torch.Tensor([0.0]).to(self.device)
        current_time = time.time()
        adv_time = current_time - start_time
        start_time = time.time()
        loss_l1 = self.l1_loss(audio, audio_recon) * self.loss_weights.l1
        loss_l2 = self.l2_loss(audio, audio_recon) * self.loss_weights.l2
        loss_kld = self.kld_loss(latent_mean, latent_log_var) * self._get_loss_weight_kld(current_step)
        loss_mel = self.mel_loss(audio, audio_recon) * self.loss_weights.mel
        loss_stft = self.stft_loss(audio, audio_recon) * self.loss_weights.stft
        loss_sisnr = self.sisnr_loss(audio, audio_recon) * self.loss_weights.sisnr
        if loss_mel.item() > last_loss_mel * 3:
            for i in range(len(batch)):
                loss_mel_i = self.mel_loss(audio[i:i + 1, ...], audio_recon[i:i + 1, ...]) * self.loss_weights.mel
                if loss_mel_i.item() > last_loss_mel * 3:
                    self.logger.warning("Possible outliers detected: song_id: %s, loss_mel = %f, last_loss_mel = %f",
                                        batch["obj_id"][i],
                                        loss_mel_i.item(), last_loss_mel)
        _, loss_gen, loss_fm = self.model_engine_d(audio, audio_recon)
        loss_gen *= self.loss_weights.gen
        loss_fm *= self.loss_weights.fm
        loss_gen_total = loss_l1 + loss_l2 + loss_mel + loss_kld + loss_stft + loss_sisnr + loss_gen + loss_fm
        loss_total = loss_adv + loss_gen_total
        current_time = time.time()
        loss_time = current_time - start_time
        start_time = time.time()
        self.model_engine_g.backward(loss_gen_total)
        self.model_engine_g.step()
        current_time = time.time()
        backward_time = current_time - start_time
        start_time = time.time()
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
        current_time = time.time()
        logging_time = current_time - start_time
        if self.rank == 0:
            self.logger.info(f"Training step {current_step}/{self.total_num_steps} "
                             f"(epoch {epoch}, step {step_in_epoch}/{epoch_length}): "
                             f"forward time = {forward_g_time},"
                             f"adversarial time = {adv_time},"
                             f"loss time = {loss_time},"
                             f"backward time = {backward_time},"
                             f"logging time = {logging_time}")
        return loss_dict, metrics_dict, loss_mel.item()

    def start_training(self):
        if self.rank == 0:
            self.logger.info(f"Start training from epoch {self.initial_epoch} ...")
        total_steps = self.initial_epoch * self.training_ds.count() // self.total_batch_size
        trn_loss_dict = OrderedDict()
        trn_metrics_dict = {
            "latent_mean_mean": torch.Tensor([0.0]).to(self.device),
            "latent_mean_std": torch.Tensor([0.0]).to(self.device),
            "latent_std_mean": torch.Tensor([0.0]).to(self.device),
            "latent_std_std": torch.Tensor([0.0]).to(self.device)
        }
        for epoch in range(self.initial_epoch, self.config.num_epochs):
            training_data_iterator, epoch_length = self.get_data_iterator(self.training_ds,
                                                                          epoch=epoch,
                                                                          shuffle=True)
            self.logger.info(f"Training data iterator got for rank {self.rank}, epoch {epoch}.")
            dist.barrier()
            last_loss_mel = 100.0
            self.skip_data = (self.initial_step > 0)
            torch_profiler_conf = self.config.torch_profiler
            if torch_profiler_conf.enabled:
                with open(self.model_engine_g.config, "r") as f:
                    tensorboard_config = json.load(f).get("tensorboard")
                tensorboard_dir = f'{tensorboard_config.get("output_path")}/{tensorboard_config.get("job_name")}'
                prof = torch.profiler.profile(
                    schedule=torch.profiler.schedule(
                        skip_first=torch_profiler_conf.schedule.skip_first,
                        wait=torch_profiler_conf.schedule.wait,
                        warmup=torch_profiler_conf.schedule.warmup,
                        active=torch_profiler_conf.schedule.active,
                        repeat=torch_profiler_conf.schedule.repeat,
                    ),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        tensorboard_dir,
                        worker_name=f"worker_{self.rank}"),
                    record_shapes=torch_profiler_conf.record_shapes,
                    profile_memory=torch_profiler_conf.profile_memory,
                    with_stack=torch_profiler_conf.with_stack,
                )
                prof.start()
            start_time = time.time()
            for step, batch in enumerate(training_data_iterator):
                if torch_profiler_conf.enabled:
                    prof.step()
                total_steps += 1
                current_time = time.time()
                loading_time = current_time - start_time
                if self.initial_step > 0:
                    if self.rank == 0:
                        self.logger.info(f"Skipping step {step}...")
                    if step <= self.initial_step:
                        continue
                    else:
                        self.initial_step = 0
                        self.skip_data = False
                if batch["audio"] is None:
                    batch["audio"] = self._load_random_audio_chunks(batch["files"], batch["durations"])
                if self.rank == 0:
                    self.logger.info("Epoch %d, step %d/%d, data loading time: %f",
                                     epoch, step, epoch_length, loading_time)
                trn_loss_dict, trn_metrics_dict, last_loss_mel = self.train_step(
                    batch=batch,
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
                start_time = time.time()
            if torch_profiler_conf.enabled:
                prof.stop()
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