import torch
import torch.nn as nn
import torch.nn.functional as F
from ama_prof_divi_common.utils import get_logger
from typing import Optional, Dict
from .dit import DiT
from .samplers import DDIMSampler
import TrainingLoss
logger = get_logger(__name__)

class Diffusion(nn.Module):
    def __init__(self, model: DiT, sampler: DDIMSampler, training_loss: Optional[TrainingLoss] = None):
        super(Diffusion, self).__init__()
        self.model = model
        self.sampler = sampler
        self.training_loss = training_loss

    def initialize_weights(self):
        self.model.initialize_weights()

    def set_inference_timesteps(self, num_inference_timesteps: int, timestep_spacing: Optional[str] = None):
        self.sampler.set_inference_timesteps(num_inference_timesteps, timestep_spacing)

    def training_step(self, *, samples: torch.Tensor, positions: Optional[torch.Tensor] = None, conditions: Optional[torch.Tensor] = None, padding_mask: Optional[torch.Tensor] = None, context: Optional[torch.Tensor] = None, context_positions: Optional[torch.Tensor] = None, context_mask: Optional[torch.Tensor] = None, noise: Optional[torch.Tensor] = None, prompt_length: int = 0, rank: int = 0, world_size: int = 1) -> Dict[str, torch.Tensor]:
        assert self.training_loss is not None, "Training loss must be provided."
        timesteps = torch.randint(0, self.sampler.num_training_timesteps, (world_size, samples.shape[0]), device=samples.device)[rank]
        if noise is None:
            noise = torch.randn((world_size, *samples.shape), device=samples.device).to(samples.dtype)[rank]
        noisy_samples = self.sampler.add_noise(samples, noise=noise, timesteps=timesteps).to(samples.dtype)
        assert prompt_length < samples.size(1), "Prompt length must be less than the number of samples."
        noisy_samples[:, :prompt_length, :] = samples[:, :prompt_length, :]
        if self.model.use_learned_variance:
            predicted_noise, predicted_logvar = self.model(noisy_samples, timesteps=timesteps, positions=positions, condition=conditions, padding_mask=padding_mask, context=context, context_positions=context_positions, context_mask=context_mask)
        else:
            predicted_noise = self.model(noisy_samples, timesteps=timesteps, positions=positions, condition=conditions, padding_mask=padding_mask, context=context, context_positions=context_positions, context_mask=context_mask)
            predicted_logvar = None
        loss_dict = self.training_loss(x_start=samples[:, prompt_length:, :], noise=noise[:, prompt_length:, :], noisy_samples=noisy_samples[:, prompt_length:, :], timesteps=timesteps, predicted_noise=predicted_noise[:, prompt_length:, :], mask=padding_mask[:, prompt_length:], predicted_log_variance=predicted_logvar[:, prompt_length:, :] if predicted_logvar is not None else None)
        return loss_dict

    @torch.inference_mode()
    def generate(self, x: torch.Tensor, *, prompt: Optional[torch.Tensor] = None, positions: Optional[torch.Tensor] = None, conditions: Optional[torch.Tensor] = None, padding_mask: Optional[torch.Tensor] = None, context: Optional[torch.Tensor] = None, context_positions: Optional[torch.Tensor] = None, context_mask: Optional[torch.Tensor] = None, start_timestep_index: int = 0, cfg_scale: Optional[float] = None, eta: float = 0.0) -> torch.Tensor:
        assert self.sampler.inference_timesteps is not None, ("Inference timesteps must be set before generating samples.")
        use_cfg = cfg_scale is not None and (conditions is not None or context is not None)
        for i in range(start_timestep_index, len(self.sampler.inference_timesteps)):
            timestep = self.sampler.inference_timesteps[i].to(x.device)
            if use_cfg:
                model_output = self.model.forward_with_cfg(x, timesteps=timestep, cfg_scale=cfg_scale, prompt=prompt, positions=positions, condition=conditions, padding_mask=padding_mask, context=context, context_mask=context_mask, context_positions=context_positions)
            else:
                model_output = self.model.forward(x, timesteps=timestep, prompt=prompt, positions=positions, condition=conditions, padding_mask=padding_mask, context=context, context_mask=context_mask, context_positions=context_positions)
            x, _, _, _ = self.sampler.denoise(model_output=model_output, sample=x, timestep_index=i, eta=eta, use_clipped_model_output=self.sampler.clip_sample_range is not None)
        return x

    @torch.inference_mode()
    def test_inference(self, samples: torch.Tensor, positions: Optional[torch.Tensor] = None, conditions: Optional[torch.Tensor] = None, padding_mask: Optional[torch.Tensor] = None, context: Optional[torch.Tensor] = None, context_positions: Optional[torch.Tensor] = None, context_mask: Optional[torch.Tensor] = None, noise: Optional[torch.Tensor] = None, prompt_length: int = 0, cfg_scale: Optional[float] = None, eta: float = 0.0, rank: int = 0, world_size: int = 1) -> torch.Tensor:
        timesteps = (torch.ones((samples.shape[0],), device=samples.device) * (self.sampler.num_training_timesteps - 1)).long()
        if noise is None:
            noise = torch.randn((world_size, *samples.shape), device=samples.device).to(samples.dtype)[rank]
        noisy_samples = self.sampler.add_noise(samples, noise=noise, timesteps=timesteps).to(samples.dtype)
        generated = self.generate(x=noisy_samples, conditions=conditions, padding_mask=padding_mask, context=context, context_mask=context_mask, cfg_scale=cfg_scale, eta=eta)
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(-1)
            generated = generated * padding_mask
            samples = samples * padding_mask
        return F.mse_loss(generated, samples)

    @torch.inference_mode()
    def shallow_diffusion(self, ref_samples: torch.Tensor, *, prompt: Optional[torch.Tensor] = None, positions: Optional[torch.Tensor] = None, conditions: Optional[torch.Tensor] = None, padding_mask: Optional[torch.Tensor] = None, context: Optional[torch.Tensor] = None, context_positions: Optional[torch.Tensor] = None, context_mask: Optional[torch.Tensor] = None, noise: Optional[torch.Tensor] = None, start_timestep_index: int = 0, cfg_scale: Optional[float] = None, eta: float = 0.0) -> torch.Tensor:
        assert self.sampler.inference_timesteps is not None, ("Inference timesteps must be set before generating samples.")
        assert 0 <= start_timestep_index < len(self.sampler.inference_timesteps), (f"Invalid start_timestep_index {start_timestep_index}. Must between 0 and len(self.sampler.inference_timesteps).")
        timesteps = torch.ones((ref_samples.shape[0],), device=ref_samples.device) * self.sampler.inference_timesteps[start_timestep_index].to(ref_samples.device)
        if noise is None:
            noise = torch.randn(ref_samples.shape, device=ref_samples.device).to(ref_samples.dtype)
        noisy_samples = self.sampler.add_noise(ref_samples, noise=noise, timesteps=timesteps).to(ref_samples.dtype)
        generated = self.generate(x=noisy_samples, prompt=prompt, positions=positions, conditions=conditions, padding_mask=padding_mask, context=context, context_mask=context_mask, start_timestep_index=start_timestep_index, cfg_scale=cfg_scale, eta=eta)
        return generated