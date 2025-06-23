import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.monitor.monitor import MonitorMaster
from deepspeed.runtime.config import DeepSpeedConfig
import ama_prof_divi_common.utils.dist_wrapper as dist
from .dataset import ChunkedWavDataset, NUM_LOCAL_RANKS
from Code_for_Experiment.Targeted_Training.hifigan_vocoder.ama-prof-divi_hifigan.mel.mel import MelGenerator
from ..model.loss import discriminator_loss, generator_loss
def __init__(self, model_engine_g: deepspeed.DeepSpeedEngine, lr_scheduler_g: Any, model_engine_d: deepspeed.DeepSpeedEngine, lr_scheduler_d: Any, mel_generator: MelGenerator):
    self.model_engine_g = model_engine_g
    self.lr_scheduler_g = lr_scheduler_g
    self.model_engine_d = model_engine_d
    self.lr_scheduler_d = lr_scheduler_d
    self.rank = dist.get_rank()
    self.local_rank = dist.get_local_rank()
    self.world_size = dist.get_world_size()
    self.device = (torch.device(get_accelerator().device_name(), self.local_rank) if (self.local_rank > -1) and get_accelerator().is_available() else torch.device("cpu"))
    self.mel_generator = mel_generator.to(self.device)
    self.logger = deepspeed.logger
    hparams = get_hparams()
    self.config = hparams.training.trainer
    ds_config = DeepSpeedConfig(model_engine_g.config)
    self.monitor = MonitorMaster(monitor_config=ds_config.monitor_config)
    self.mini_batch_size = self.model_engine_g.train_micro_batch_size_per_gpu()
    self.total_batch_size = self.mini_batch_size * self.world_size
    self.training_ds, self.validation_ds = ChunkedWavDataset.make_datasets()
    self.logger.info(f"Training dataset: {len(self.training_ds)} items")
    self.logger.info(f"Validation dataset: {len(self.validation_ds)} items.")
    self.training_sampler = DistributedSampler(self.training_ds, num_replicas=NUM_LOCAL_RANKS, shuffle=True, seed=self.config.random_seed, rank=self.local_rank)
    self.training_dataloader = DataLoader(self.training_ds, batch_size=self.mini_batch_size, sampler=self.training_sampler, num_workers=self.config.dataloader_num_workers, collate_fn=self.training_ds.collate_fn)
    self.validation_sampler = DistributedSampler(self.validation_ds, num_replicas=NUM_LOCAL_RANKS, shuffle=True, seed=self.config.random_seed, rank=self.local_rank)
    self.validation_dataloader = DataLoader(self.validation_ds, batch_size=self.mini_batch_size, sampler=self.validation_sampler, num_workers=self.config.dataloader_num_workers, collate_fn=self.validation_ds.collate_fn)
    self.total_num_steps = len(self.training_dataloader) * self.config.num_epochs
    self.lr_scheduler_g.total_num_steps = self.total_num_steps
    self.lr_scheduler_d.total_num_steps = self.total_num_steps
    self.discriminator_warmup_steps = self.config.discriminator_warmup_steps
    self.initial_epoch = 0
    self.initial_step = 0
    self.load_checkpoint()
def close(self):
    self.training_ds.close()
    self.validation_ds.close()
def __enter__(self):
    pass
def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()
def load_checkpoint(self):
    if self.rank == 0:
        self.logger.info(f"Trying to load checkpoint from {self.config.checkpoint_dir} ...")
    ckpt_dir = Path(self.config.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True, parents=False)
    ckpt_dir_g = ckpt_dir / "g"
    ckpt_dir_d = ckpt_dir / "d"
    ckpt_dir_g.mkdir(exist_ok=True, parents=False)
    ckpt_dir_d.mkdir(exist_ok=True, parents=False)
    _, client_state = self.model_engine_g.load_checkpoint(load_dir=str(ckpt_dir_g), load_module_strict=True, load_optimizer_states=True, load_lr_scheduler_states=True)
    self.model_engine_d.load_checkpoint(load_dir=str(ckpt_dir_d), load_module_strict=True, load_optimizer_states=True, load_lr_scheduler_states=True)
    if client_state is not None:
        self.initial_epoch = client_state["current_epoch"]
        self.initial_step = client_state["current_step"]
        if self.rank == 0:
            self.logger.info("Checkpoint loaded. Epoch = %d, step = %d", self.initial_epoch, self.initial_step)
    dist.barrier()
def save_checkpoint(self, epoch: int, step: int, total_steps: int):
    ckpt_dir = Path(self.config.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True, parents=False)
    ckpt_dir_g = ckpt_dir / "g"
    ckpt_dir_d = ckpt_dir / "d"
    ckpt_dir_g.mkdir(exist_ok=True, parents=False)
    ckpt_dir_d.mkdir(exist_ok=True, parents=False)
    client_state = {
        "current_epoch": epoch,
        "current_step": step,
        "current_total_steps": total_steps,
    }
    self.model_engine_g.save_checkpoint(save_dir=str(ckpt_dir_g), client_state=client_state)
    self.model_engine_d.save_checkpoint(save_dir=str(ckpt_dir_d), client_state=client_state)
    if self.rank == 0:
        self.logger.info("Checkpoints saved. Epoch = %d, step = %d", epoch, step)
def validate_loop(self, training_loss_dict: OrderedDict[str, torch.Tensor], current_step: int):
    for key in training_loss_dict:
        dist.all_reduce(training_loss_dict[key], op=dist.ReduceOp.SUM)
        training_loss_dict[key] /= self.world_size
    if self.rank == 0:
        self.logger.info(f"Validation at step {current_step} ...")
    self.model_engine_g.eval()
    self.model_engine_d.eval()
    step = 0
    loss_dict = OrderedDict({
        "val_loss_disc": torch.Tensor([0.0]).to(self.device),
        "val_loss_gen": torch.Tensor([0.0]).to(self.device),
        "val_loss_fm": torch.Tensor([0.0]).to(self.device),
        "val_loss_mel": torch.Tensor([0.0]).to(self.device),
        "val_loss_gen_total": torch.Tensor([0.0]).to(self.device),
        "val_loss_total": torch.Tensor([0.0]).to(self.device)
    })
    loss_dict = OrderedDict(**training_loss_dict, **loss_dict)
    with torch.no_grad():
        for step, batch in enumerate(self.validation_dataloader):
            audio = batch["wav"]["data"].to(self.device)
            mel = batch["mel"]["data"].to(self.device)
            audio_pred = self.model_engine_g(mel)
            loss_disc, loss_gen, loss_fm = self.model_engine_d(audio, audio_pred)
            audio_pred_mel = self.mel_generator(audio_pred).squeeze(1)
            loss_mel = F.l1_loss(mel, audio_pred_mel)
            loss_gen_total = loss_gen + loss_fm + (loss_mel * self.config.mel_loss_weight)
            loss_total = loss_disc + loss_gen_total
            loss_dict["val_loss_disc"] += loss_disc
            loss_dict["val_loss_gen"] += loss_gen
            loss_dict["val_loss_fm"] += loss_fm
            loss_dict["val_loss_mel"] += loss_mel
            loss_dict["val_loss_gen_total"] += loss_gen_total
            loss_dict["val_loss_total"] += loss_total
            if step > self.config.num_validation_steps:
                break
    scale = (step + 1) * self.world_size
    for key in loss_dict:
        dist.all_reduce(loss_dict[key], op=dist.ReduceOp.SUM)
        loss_dict[key] /= scale
    if self.rank == 0:
        self.logger.info(f"Validation at step {current_step}/{self.total_num_steps}: loss_disc = {loss_dict['val_loss_disc'].item()}, loss_gen = {loss_dict['val_loss_gen_total'].item()}, loss_total = {loss_dict['val_loss_total'].item()}")
    global_steps = self.model_engine_g.global_samples // self.total_batch_size
    event_list = [
        ("Validation/" + key, value, global_steps)
        for key, value in loss_dict.items()
    ]
    self.monitor.write_events(event_list)
def train_step(self, batch: Dict, current_step: int, epoch: int, step_in_epoch: int) -> OrderedDict[str, torch.Tensor]:
    self.model_engine_g.train()
    self.model_engine_d.train()
    self.model_engine_g.zero_grad()
    self.model_engine_d.zero_grad()
    audio = batch["wav"]["data"].to(self.device)
    mel = batch["mel"]["data"].to(self.device)
    audio_pred = self.model_engine_g(mel)
    if current_step > self.discriminator_warmup_steps:
        loss_disc, _, _ = self.model_engine_d(audio, audio_pred.detach())
        self.model_engine_d.backward(loss_disc)
        self.model_engine_d.step()
    else:
        loss_disc = torch.Tensor([0.0]).to(self.device)
    audio_pred_mel = self.mel_generator(audio_pred).squeeze(1)
    loss_mel = F.l1_loss(mel, audio_pred_mel)
    if self.rank == 0:
        self.logger.info("mel: %s, min=%f, max=%f", mel.shape, mel.min().item(), mel.max().item())
        self.logger.info("audio_pred_mel: %s, min=%f, max=%f", audio_pred_mel.shape, audio_pred_mel.min().item(), audio_pred_mel.max().item())
        self.logger.info("loss_mel: %f", loss_mel.item())
    _, loss_gen, loss_fm = self.model_engine_d(audio, audio_pred)
    loss_gen_total = loss_gen + loss_fm + (loss_mel * self.config.mel_loss_weight)
    loss_total = loss_disc + loss_gen_total
    self.model_engine_g.backward(loss_gen_total)
    self.model_engine_g.step()
    if self.rank == 0:
        self.logger.info(f"Training step {current_step}/{self.total_num_steps} (epoch {epoch}, step {step_in_epoch}): loss_disc = {loss_disc.item()}, loss_gen = {loss_gen_total.item()}, loss_mel = {loss_mel.item()}, loss_total = {loss_total.item()}")
    global_steps = self.model_engine_g.global_samples // self.total_batch_size
    self.monitor.write_events([
        ("Train/loss_disc", loss_disc, global_steps),
        ("Train/loss_gen", loss_gen, global_steps),
        ("Train/loss_fm", loss_fm, global_steps),
        ("Train/loss_mel", loss_mel, global_steps),
        ("Train/loss_gen_total", loss_gen_total, global_steps),
        ("Train/loss_total", loss_total, global_steps)
    ])
    return {
        "trn_loss_disc": loss_disc,
        "trn_loss_gen": loss_gen,
        "trn_loss_fm": loss_fm,
        "trn_loss_mel": loss_mel,
        "trn_loss_gen_total": loss_gen_total,
        "trn_loss_total": loss_total
    }
def start_training(self):
    if self.rank == 0:
        self.logger.info(f"Start training from epoch {self.initial_epoch} ...")
    total_steps = self.initial_epoch * len(self.training_dataloader)
    trn_loss_dict = OrderedDict()
    for epoch in range(self.initial_epoch, self.config.num_epochs):
        self.training_sampler.set_epoch(epoch)
        self.validation_sampler.set_epoch(epoch)
        for step, batch in enumerate(self.training_dataloader):
            total_steps += 1
            if self.initial_step > 0:
                if self.rank == 0:
                    self.logger.info(f"Skipping step {step} ...")
                if step < self.initial_step:
                    continue
                self.initial_step = 0
            trn_loss_dict = self.train_step(batch, total_steps, epoch, step)
            if total_steps % self.config.validation_interval == 0:
                self.validate_loop(training_loss_dict=trn_loss_dict, current_step=total_steps)
                deepspeed.comm.log_summary()
            if total_steps % self.config.checkpoint_interval == 0:
                self.save_checkpoint(epoch=epoch, step=step, total_steps=total_steps)
        self.validate_loop(training_loss_dict=trn_loss_dict, current_step=total_steps)
        self.save_checkpoint(epoch=epoch + 1, step=0, total_steps=total_steps)
        deepspeed.comm.log_summary()
    dist.barrier()
    if self.rank == 0:
        self.logger.info("Done.")