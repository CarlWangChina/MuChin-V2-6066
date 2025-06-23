import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm
from ama_prof_divi.utils.logging import get_logger
from .model import UnetDiffusionModel, WaveNetDiffusionModel
from .sampler_list import get_sampler
from .unet import UnetModelArgs
from .wavenet import WaveNetModelArgs

logger = get_logger(__name__)

@dataclass
class DiffusionArgs:
    sampler: str = None
    sampler_extra_args: dict = None
    denoiser: str = None
    unet: Optional[UnetModelArgs] = None
    wavenet: Optional[WaveNetModelArgs] = None
    guidance_scale: float = 7.5

    @property
    def in_channels(self):
        if self.denoiser == "unet":
            return self.unet.in_channels
        elif self.denoiser == "wavenet":
            return self.wavenet.in_channels
        else:
            raise ValueError("Unknown denoiser: {}".format(self.denoiser))

    @property
    def model_channels(self):
        if self.denoiser == "unet":
            return self.unet.model_channels
        elif self.denoiser == "wavenet":
            return self.wavenet.model_channels
        else:
            raise ValueError("Unknown denoiser: {}".format(self.denoiser))

    @property
    def context_dim(self):
        if self.denoiser == "unet":
            return self.unet.context_dim
        elif self.denoiser == "wavenet":
            return self.wavenet.context_channels
        else:
            raise ValueError("Unknown denoiser: {}".format(self.denoiser))

class LatentDiffusion(nn.Module):
    def __init__(self, args: DiffusionArgs, *, training: bool = False, device: str = "cpu"):
        super(LatentDiffusion, self).__init__()
        self.args = args
        self.denoiser = args.denoiser
        if self.denoiser == "unet":
            assert args.unet is not None, "Must specify U-Net arguments."
            self.model = UnetDiffusionModel(args.unet, device=device)
        elif self.denoiser == "wavenet":
            assert args.wavenet is not None, "Must specify WaveNet arguments."
            self.model = WaveNetDiffusionModel(args.wavenet, device=device)
        else:
            raise ValueError("Unknown denoiser: {}".format(args.denoiser))
        self.sampler = get_sampler(name=args.sampler, args=args.sampler_extra_args, training=training, device=device)
        self.criteria = nn.MSELoss()
        logger.info("Diffusion sampler is: {}".format(self.sampler.name))

    def freeze_unet_model(self, freeze: bool = True):
        if self.denoiser == "unet":
            self.model.freeze_unet_model(freeze)
        else:
            logger.warning("Ignored freeze_unet_model() because denoiser is not U-Net.")

    @property
    def device(self):
        return self.sampler.device

    @torch.inference_mode()
    def generate(self, *, seq_len: int, context: Optional[torch.Tensor] = None, condition: Optional[torch.Tensor] = None, latent_start: Optional[torch.Tensor] = None, description: str = None) -> torch.Tensor:
        time_steps = self.sampler.time_steps
        if context is not None:
            assert context.dim() == 3, "Context must be a 3D tensor"
            assert context.shape[1] == self.args.context_dim, "Context dimension does not match model context dimension"
            batch_size = context.shape[0]
        else:
            batch_size = 1
        if description is None:
            description = "Generating by diffusion"
        if latent_start is None:
            latents = torch.randn(batch_size, self.args.in_channels, seq_len, device=self.device)
        else:
            assert latent_start.shape == (batch_size, self.args.in_channels, seq_len), "Noise tensor shape does not match model input shape."
            latents = latent_start.clone().to(device=self.device)
        latents = latents * self.sampler.init_noise_sigma
        if context is not None:
            logger.info(f"Context: {context.shape}, {context.device}")
            no_context = torch.zeros(context.shape, device=self.device)
            context_input = torch.cat([no_context, context])
            condition_input = torch.cat([condition] * 2) if condition is not None else None
            batch_size *= 2
        else:
            context_input = None
            condition_input = condition
        states = {}
        for t in tqdm(time_steps, desc=description):
            if context is not None:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents
            latent_model_input = self.sampler.scale_model_input(latent_model_input, time_step=t)
            with torch.no_grad():
                noise_pred = self.model(latent_model_input, time_steps=torch.IntTensor([t] * batch_size).to(device=self.device), condition=condition_input, context=context_input)
            if context is not None:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + self.args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            latents = self.sampler.sample(model_output=noise_pred, time_step=t, sample=latents, states=states)["prev_sample"]
        return latents

    def forward(self, *, latent: torch.Tensor, context: Optional[torch.Tensor], condition: Optional[torch.Tensor] = None) -> dict:
        assert latent.dim() == 3, "Latent must be a 3D tensor"
        assert latent.shape[1] == self.args.in_channels, (f"Latent channel dimension ({latent.shape[1]}) does not match model input channel dimension ({self.args.in_channels}).")
        batch_size = latent.shape[0]
        noise = torch.randn(latent.shape, device=latent.device, dtype=next(self.parameters()).dtype)
        time_steps = torch.randint(0, self.sampler.num_training_steps, (batch_size,), device=latent.device)
        logger.debug("time_steps: %s", time_steps)
        noisy_latent = self.sampler.add_noise(latent, noise, time_steps)
        noise_pred = self.model(noisy_latent, time_steps=time_steps, condition=condition, context=context)
        loss = self.criteria(noise_pred, noise)
        return {
            "time_steps": time_steps,
            "noisy_latent": noisy_latent,
            "noise": noise,
            "noise_pred": noise_pred,
            "loss": loss
        }