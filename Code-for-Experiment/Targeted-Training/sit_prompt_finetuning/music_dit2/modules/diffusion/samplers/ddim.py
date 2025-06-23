import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger
from .utils import (
    make_beta_schedule,
    update_time_steps,
    threshold_sample,
    extract_into_tensor
)

logger = get_logger(__name__)

class DDIMSampler(nn.Module):
    def __init__(self,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02,
                 beta_schedule: str = "linear",
                 timestep_spacing: str = "leading",
                 num_training_timesteps: int = 1000,
                 dynamic_thresholding_ratio: Optional[float] = None,
                 clip_sample_range: Optional[Tuple[float, float]] = None,
                 device: Optional[torch.device] = None):
        super(DDIMSampler, self).__init__()
        self.num_training_timesteps = num_training_timesteps
        self.timestep_spacing = timestep_spacing
        self.dynamic_thresholding_ratio: Optional[float] = dynamic_thresholding_ratio
        self.clip_sample_range: Optional[Tuple[float, float]] = clip_sample_range
        self.inference_timesteps = None
        self.register_buffer("betas",
                             make_beta_schedule(schedule=beta_schedule,
                                                num_steps=num_training_timesteps,
                                                beta_start=beta_start,
                                                beta_end=beta_end))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("sqrt_alphas_cumprod", self.alphas_cumprod ** 0.5)
        self.register_buffer("sqrt_betas_cumprod", (1.0 - self.alphas_cumprod) ** 0.5)
        self.register_buffer("alphas_cumprod_prev",
                             torch.cat([torch.Tensor([1.0]).to(self.alphas_cumprod.device),
                                        self.alphas_cumprod[:-1]]))
        self.register_buffer("posterior_variance",
                             self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.register_buffer("posterior_log_variance_clipped", torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]]) * 2.0
            if len(self.posterior_variance) > 1 else torch.Tensor([])
        ))
        self.register_buffer("posterior_mean_coeff1",
                             self.betas * (self.alphas_cumprod_prev ** 0.5) / (1.0 - self.alphas_cumprod))
        self.register_buffer("posterior_mean_coeff2",
                             (1.0 - self.alphas_cumprod_prev) * (self.alphas ** 0.5) / (1.0 - self.alphas_cumprod))
        if device is not None:
            self.to(device)

    def set_inference_timesteps(self,
                                num_inference_timesteps: int,
                                timestep_spacing: Optional[str] = None):
        if timestep_spacing is None:
            timestep_spacing = self.timestep_spacing
        assert num_inference_timesteps <= self.num_training_timesteps, \
            (f"num_inference_timesteps ({num_inference_timesteps}) must be less than or equal to "
             f"num_training_timesteps ({self.num_training_timesteps})")
        self.inference_timesteps = update_time_steps(num_training_steps=self.num_training_timesteps,
                                                    num_inference_steps=num_inference_timesteps,
                                                    timestep_spacing=timestep_spacing).to(self.device)

    @property
    def device(self):
        return self.betas.device

    @property
    def num_inference_timesteps(self) -> Optional[int]:
        return len(self.inference_timesteps) if self.inference_timesteps is not None else None

    def add_noise(self,
                  x: torch.Tensor,
                  noise: torch.Tensor,
                  timesteps: torch.Tensor) -> torch.Tensor:
        assert x.size() == noise.size(), \
            f"The size of x ({x.size()}) and noise ({noise.size()}) should be the same."
        timesteps = timesteps.long()
        assert timesteps.dim() == 1, \
            f"The timesteps should be a 1D tensor, but got {timesteps.dim()}D."
        assert x.size(0) == timesteps.size(0), \
            f"The batch size of x ({x.size(0)}) and timesteps ({timesteps.size(0)}) should be the same."
        assert 0 <= timesteps.min() <= timesteps.max() < self.num_training_timesteps, \
            (f"The timesteps should be in the range of [0, {self.num_training_timesteps - 1}], "
             f"but got {timesteps.min()} to {timesteps.max()}.")
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod[timesteps].flatten()
        sqrt_betas_cumprod = self.sqrt_betas_cumprod[timesteps].flatten()
        while sqrt_alphas_cumprod.dim() < x.dim():
            sqrt_alphas_cumprod = sqrt_alphas_cumprod.unsqueeze(-1)
            sqrt_betas_cumprod = sqrt_betas_cumprod.unsqueeze(-1)
        noisy_samples = sqrt_alphas_cumprod * x + sqrt_betas_cumprod * noise
        return noisy_samples

    def q_posterior_mean_variance(self,
                                  samples: torch.Tensor,
                                  noisy_samples: torch.Tensor,
                                  timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        posterior_mean = (extract_into_tensor(self.posterior_mean_coeff1,
                                              timesteps,
                                              samples.shape) * samples
                         + extract_into_tensor(self.posterior_mean_coeff2,
                                               timesteps,
                                               samples.shape) * noisy_samples)
        posterior_log_variance = extract_into_tensor(self.posterior_log_variance_clipped,
                                                    timesteps,
                                                    samples.shape)
        return posterior_mean, posterior_log_variance

    def denoise(self,
                *,
                model_output: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                sample: torch.Tensor,
                timestep_index: Optional[int] = None,
                timesteps: Optional[torch.Tensor] = None,
                eta: float = 0.0,
                use_clipped_model_output: bool = False,
                variance_noise: Optional[torch.Tensor] = None,
                generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if timesteps is not None:
            assert timestep_index is None, "timestep_index must be None if timesteps tensor is provided."
            current_timestep = timesteps
            prev_timestep = torch.clamp(current_timestep - 1, min=0).long()
        else:
            assert self.inference_timesteps is not None, \
                "Please call set_inference_timesteps() to set inference time step before running the prediction step."
            assert 0 <= timestep_index < len(self.inference_timesteps), \
                f"timestep_index ({timestep_index}) should be in the range of [0, {len(self.inference_timesteps) - 1}]."
            current_timestep = self.inference_timesteps[timestep_index]
            prev_timestep = self.inference_timesteps[timestep_index + 1] \
                if timestep_index + 1 < len(self.inference_timesteps) else None
        sqrt_betas_cumprod = (extract_into_tensor(self.sqrt_betas_cumprod, current_timestep, sample.shape)
                              .to(sample.dtype))
        sqrt_alphas_cumprod = (extract_into_tensor(self.sqrt_alphas_cumprod, current_timestep, sample.shape)
                               .to(sample.dtype))
        alphas_cumprod_prev = extract_into_tensor(self.alphas_cumprod, prev_timestep, sample.shape).to(sample.dtype) \
            if prev_timestep is not None else 1.0
        if isinstance(model_output, tuple):
            model_output, log_var = model_output
            min_log_variance = extract_into_tensor(self.posterior_log_variance_clipped,
                                                   current_timestep,
                                                   sample.shape).to(sample.dtype)
            max_log_variance = torch.log(extract_into_tensor(self.betas,
                                                             current_timestep,
                                                             sample.shape).to(sample.dtype))
            log_var = 0.5 * (log_var + 1) * (max_log_variance - min_log_variance) + min_log_variance
            variance = torch.exp(log_var)
        else:
            variance = extract_into_tensor(self.posterior_variance,
                                           current_timestep,
                                           sample.shape).to(sample.dtype)
            log_var = extract_into_tensor(self.posterior_log_variance_clipped,
                                          current_timestep,
                                          sample.shape).to(sample.dtype)
        pred_original_sample = (sample - sqrt_betas_cumprod * model_output) / sqrt_alphas_cumprod
        if self.dynamic_thresholding_ratio is not None:
            pred_original_sample = threshold_sample(sample=pred_original_sample,
                                                    dynamic_thresholding_ratio=self.dynamic_thresholding_ratio)
        elif self.clip_sample_range is not None:
            pred_original_sample = torch.clamp(pred_original_sample,
                                               self.clip_sample_range[0],
                                               self.clip_sample_range[1])
        pred_epsilon = model_output
        sigma = eta * torch.sqrt(variance)
        if use_clipped_model_output:
            pred_epsilon = (sample - sqrt_alphas_cumprod * pred_original_sample) / sqrt_betas_cumprod
        pred_sample_direction = ((1 - alphas_cumprod_prev - sigma**2) ** 0.5) * pred_epsilon
        prev_sample = (alphas_cumprod_prev ** 0.5) * pred_original_sample + pred_sample_direction
        if eta > 0:
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )
            if variance_noise is None:
                variance_noise = torch.randn(model_output.shape,
                                             generator=generator,
                                             device=model_output.device,
                                             dtype=model_output.dtype).clamp(-3.0, 3.0)
            variance = sigma * variance_noise
            prev_sample = prev_sample + variance
        return prev_sample, pred_original_sample, variance, log_var