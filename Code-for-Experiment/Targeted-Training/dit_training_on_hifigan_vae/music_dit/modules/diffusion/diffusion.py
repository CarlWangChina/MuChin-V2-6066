import torch
import torch.nn as nn
from music_dit.utils import get_logger
from typing import Optional
from .dit import DiT
from .samplers import DDIMSampler
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.modules.diffusion.losses.training_loss import TrainingLoss

logger = get_logger(__name__)

class Diffusion(nn.Module):
    def __init__(self,
                 model: DiT,
                 sampler: DDIMSampler,
                 training_loss: Optional[TrainingLoss] = None):
        super().__init__()
        self.model = model
        self.sampler = sampler
        self.training_loss = training_loss

    def set_inference_timesteps(self,
                                num_inference_timesteps: int,
                                timestep_spacing: Optional[str] = None):
        self.sampler.set_inference_timesteps(num_inference_timesteps, timestep_spacing)

    def training_step(self,
                      *,
                      samples: torch.Tensor,
                      prompt: Optional[torch.Tensor] = None,
                      positions: Optional[torch.Tensor] = None,
                      conditions: Optional[torch.Tensor] = None,
                      padding_mask: Optional[torch.Tensor] = None,
                      context: Optional[torch.Tensor] = None,
                      context_positions: Optional[torch.Tensor] = None,
                      context_mask: Optional[torch.Tensor] = None,
                      noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert self.training_loss is not None, "Training loss must be provided."
        timesteps = torch.randint(0,
                                  self.sampler.num_training_timesteps,
                                  (samples.shape[0],),
                                  device=samples.device)
        if noise is None:
            noise = torch.randn(samples.shape, device=samples.device)
        noisy_samples = self.sampler.add_noise(samples,
                                               noise=noise,
                                               timesteps=timesteps)
        if self.model.use_learned_variance:
            predicted_noise, predicted_var = self.model(noisy_samples,
                                                        timesteps=timesteps,
                                                        prompt=prompt,
                                                        positions=positions,
                                                        condition=conditions,
                                                        padding_mask=padding_mask,
                                                        context=context,
                                                        context_positions=context_positions,
                                                        context_mask=context_mask)
        else:
            predicted_noise = self.model(noisy_samples,
                                         timesteps=timesteps,
                                         prompt=prompt,
                                         positions=positions,
                                         condition=conditions,
                                         padding_mask=padding_mask,
                                         context=context,
                                         context_positions=context_positions,
                                         context_mask=context_mask)
            predicted_var = None
        loss = self.training_loss(x_start=samples,
                                  noise=noise,
                                  noisy_samples=noisy_samples,
                                  predicted_noise=predicted_noise,
                                  timesteps=timesteps,
                                  predicted_log_variance=predicted_var)
        return loss

    @torch.inference_mode()
    def generate(self,
                 x: torch.Tensor,
                 *,
                 prompt: Optional[torch.Tensor] = None,
                 positions: Optional[torch.Tensor] = None,
                 conditions: Optional[torch.Tensor] = None,
                 context: Optional[torch.Tensor] = None,
                 context_positions: Optional[torch.Tensor] = None,
                 context_mask: Optional[torch.Tensor] = None,
                 start_timestep_index: int = 0,
                 cfg_scale: Optional[float] = None,
                 eta: float = 0.0) -> torch.Tensor:
        assert self.sampler.inference_timesteps is not None, "Inference timesteps must be set before generating samples."
        use_cfg = cfg_scale is not None and (conditions is not None or context is not None)
        for i in range(start_timestep_index, len(self.sampler.inference_timesteps)):
            timestep = self.sampler.inference_timesteps[i]
            if use_cfg:
                model_output = self.model.forward_with_cfg(
                    x,
                    timesteps=timestep,
                    cfg_scale=cfg_scale,
                    prompt=prompt,
                    positions=positions,
                    condition=conditions,
                    context=context,
                    context_mask=context_mask,
                    context_positions=context_positions)
            else:
                model_output = self.model.forward(
                    x,
                    timesteps=timestep,
                    prompt=prompt,
                    positions=positions,
                    condition=conditions,
                    context=context,
                    context_mask=context_mask,
                    context_positions=context_positions)
            x, _ = self.sampler.denoise(model_output=model_output,
                                        timestep_index=i,
                                        sample=x,
                                        eta=eta,
                                        use_clipped_model_output=self.sampler.clip_sample_range is not None)
        return x