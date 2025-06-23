import torch
import torch.nn as nn
import torch.distributed as dist
from pathlib import Path
from einops import rearrange
from typing import Optional
from ama_prof_divi.utils import download_file
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger
from ama_prof_divi.modules.diffusion import UnetModelArgs, DiffusionArgs, WaveNetModelArgs, LatentDiffusion

logger = get_logger(__name__)

class DiffusionModel(nn.Module):
    def __init__(self, hparams: dict, extra_args: Optional[dict] = None, local_rank: int = 0, parallel_enabled: bool = False, device: torch.device or str = "cpu"):
        super(DiffusionModel, self).__init__()
        self.hparams = hparams
        self.local_rank = local_rank
        self.parallel_enabled = parallel_enabled
        diffusion_args = hparams["ama-prof-divi"]["models"]["acoustic"]["diffusion"]
        denoiser_type = diffusion_args["denoiser"]
        assert denoiser_type in ["unet", "wavenet"]
        self.dim = diffusion_args["model_channels"]
        self.vocab_size = extra_args["vocab_size"]
        self.start_id = extra_args["start_id"]
        self.pad_id = extra_args["pad_id"]
        self.prompt_embedding_dim = diffusion_args["prompt_dim"]
        self.diffusion_embedding = nn.Embedding(self.vocab_size, self.dim, padding_idx=self.pad_id, device=device)
        self.ln = nn.LayerNorm(self.dim, eps=1e-4, device=device)
        self.ln_context = nn.LayerNorm(self.prompt_embedding_dim, eps=1e-4, device=device)
        if denoiser_type == "unet":
            unet_args = UnetModelArgs(
                in_channels=self.dim,
                out_channels=self.dim,
                model_channels=self.dim,
                context_dim=diffusion_args["prompt_dim"],
                num_res_blocks=diffusion_args["unet"]["num_res_blocks"],
                attention_resolutions=diffusion_args["unet"]["attention_resolutions"],
                dropout=diffusion_args["unet"]["dropout"],
                channel_mult=diffusion_args["unet"]["channel_mult"],
                conv_resample=diffusion_args["unet"]["conv_resample"],
                dims=1,
                num_heads=diffusion_args["unet"]["num_heads"],
                use_transformer=diffusion_args["unet"]["use_transformer"],
                transformer_depth=diffusion_args["unet"]["transformer_depth"],
                use_scale_shift_norm=diffusion_args["unet"]["use_scale_shift_norm"],
                res_block_updown=diffusion_args["unet"]["res_block_updown"],
                use_time_embedding=diffusion_args["unet"]["use_time_embedding"],
                use_controlnet=False
            )
            diffusion_args = DiffusionArgs(
                sampler=diffusion_args["sampler"]["name"],
                sampler_extra_args=diffusion_args["sampler"],
                denoiser=diffusion_args["denoiser"],
                unet=unet_args,
                guidance_scale=diffusion_args["guidance_scale"]
            )
        else:
            wavenet_args = WaveNetModelArgs(
                in_channels=self.dim,
                out_channels=self.dim,
                model_channels=self.dim,
                context_channels=diffusion_args["prompt_dim"],
                num_layers=diffusion_args["wavenet"]["num_layers"],
                dilation_cycle=diffusion_args["wavenet"]["dilation_cycle"]
            )
            diffusion_args = DiffusionArgs(
                sampler=diffusion_args["sampler"]["name"],
                sampler_extra_args=diffusion_args["sampler"],
                denoiser=diffusion_args["denoiser"],
                wavenet=wavenet_args,
                guidance_scale=diffusion_args["guidance_scale"]
            )
        self.diffusion = LatentDiffusion(diffusion_args, training=True, device=device)
        if local_rank == 0 and denoiser_type == "unet":
            unet_block_desc = self.diffusion.model.unet.block_desc
            for block in unet_block_desc:
                logger.info("UNET %s", block)
        if extra_args["embedding_pretrained_model"] is not None:
            self._load_pretrained_embedding(extra_args)

    def _load_pretrained_embedding(self, extra_args: dict):
        root_path = Path(self.hparams["ama-prof-divi"]["root_path"])
        checkpoints_dir = root_path.joinpath("checkpoints").joinpath("acoustic").joinpath("diffusion")
        checkpoints_file = checkpoints_dir.joinpath(extra_args["embedding_pretrained_model"])
        if self.local_rank == 0:
            if not checkpoints_dir.exists():
                logger.info(f"Creating directory '{checkpoints_dir}' ...")
                checkpoints_dir.mkdir(parents=True)
            logger.info(f"Downloading pretrained checkpoints '{extra_args['embedding_pretrained_model']}' ...")
            checksum = None
            if "embedding_ckpt_sha256" in extra_args:
                checksum = extra_args["embedding_ckpt_sha256"]
            download_file(extra_args["embedding_ckpt_url"], str(checkpoints_file), expected_sha256=checksum)
        if self.parallel_enabled:
            dist.barrier()
        if self.local_rank == 0:
            logger.info("Loading the pretrained embedding model ...")
        state_dict = torch.load(checkpoints_file, map_location="cpu")["state_dict"]
        state_dict = {
            "weight": state_dict["embedding.weight"]
        }
        self.diffusion_embedding.load_state_dict(state_dict)

    def forward(self, *, acoustic_tokens: torch.Tensor, prompt_embedding: Optional[torch.Tensor] = None) -> dict:
        assert acoustic_tokens.dim() == 2
        if prompt_embedding is not None:
            assert prompt_embedding.dim() == 3
            assert prompt_embedding.shape == (acoustic_tokens.shape[0], 1, self.prompt_embedding_dim)
        latent = self.diffusion_embedding(acoustic_tokens)
        latent = self.ln(latent)
        latent = rearrange(latent, "b l d -> b d l")
        prompt_embedding = rearrange(self.ln_context(prompt_embedding), "b l c -> b c l") if prompt_embedding is not None else None
        result = self.diffusion(latent=latent, context=prompt_embedding)
        return {
            "loss": result["loss"],
            "latent": latent,
            "time_steps": result["time_steps"],
            "noisy_latent": result["noisy_latent"],
            "noise": result["noise"],
            "context": prompt_embedding,
            "noise_pred": result["noise_pred"]
        }