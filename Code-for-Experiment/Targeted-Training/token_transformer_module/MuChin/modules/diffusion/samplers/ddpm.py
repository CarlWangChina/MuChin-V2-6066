import torch
from typing import Optional, Union, List
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.logging import get_logger
logger = get_logger(__name__)
from .utils import threshold_sample, randn_tensor
from .sampler_base import Sampler
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.modules.diffusion.samplers.noise import add_noise_ddpm_ddim
from .consts import *

class DDPMSampler(Sampler):
    def __init__(self, name: str, args: dict = None, training: bool = False, device: str = 'cpu'):
        super(DDPMSampler, self).__init__(name=name, args=args, training=training, device=device)
        self.variance_type = self.args.get('variance_type', DDPM_VARIANCE_TYPE)
        self.prediction_type = self.args.get('prediction_type', DDPM_PREDICTION_TYPE)
        self.thresholding = self.args.get('thresholding', DDPM_THRESHOLDING)
        self.clip_sample = self.args.get('clip_sample', DDPM_CLIP_SAMPLE)
        self.clip_sample_range = self.args.get('clip_sample_range', DEFAULT_CLIP_SAMPLE_RANGE)
        self.dynamic_thresholding_ratio = self.args.get('dynamic_thresholding_ratio', DEFAULT_DYNAMIC_THRESHOLDING_RATIO)
        if not training:
            self.update_time_steps(self.num_inference_steps)

    def _get_variance(self, t, predicted_variance=None, variance_type=None):
        prev_t = self._previous_timestep(t)
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        variance = torch.clamp(variance, min=1e-4)
        if variance_type is None:
            variance_type = self.variance_type
        if variance_type == "fixed_small":
            variance = variance
        elif variance_type == "fixed_small_log":
            variance = torch.log(variance)
            variance = torch.exp(0.5 * variance)
        elif variance_type == "fixed_large":
            variance = current_beta_t
        elif variance_type == "fixed_large_log":
            variance = torch.clamp(torch.log(current_beta_t), min=0)
        elif variance_type == "learned":
            assert predicted_variance is not None
            return torch.clamp(predicted_variance, min=0.0)
        elif variance_type == "learned_range":
            assert predicted_variance is not None
            min_log = torch.log(variance)
            max_log = torch.log(torch.clamp(current_beta_t, min=1e-4))
            frac = (predicted_variance + 1) / 2
            variance = torch.clamp(frac * max_log + (1 - frac) * min_log, min=0.0, max=1.0)
        else:
            raise ValueError(f"variance_type given as {variance_type} must be one of `fixed_small`, `fixed_small_log`, `fixed_large`, `fixed_large_log`, `learned`, or `learned_range` for the DDPMScheduler.")
        return variance

    def _previous_timestep(self, timestep):
        num_inference_steps = self.num_inference_steps if self.num_inference_steps else self.num_training_steps
        prev_t = timestep - self.num_training_steps // num_inference_steps
        return prev_t

    def sample(self, model_output: torch.Tensor, time_step: int, sample: torch.Tensor, *, eta: float = 0.0, use_clipped_model_output: bool = False, generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None, variance_noise: Optional[torch.Tensor] = None, states: dict = None) -> dict:
        t = time_step
        prev_t = self._previous_timestep(t)
        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
        else:
            raise ValueError(f"prediction_type given as {self.prediction_type} must be one of `epsilon`, `sample` or `v_prediction`  for the DDPMScheduler.")
        if self.thresholding:
            pred_original_sample = threshold_sample(pred_original_sample, self.dynamic_thresholding_ratio, self.clip_sample_range)
        elif self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-self.clip_sample_range, self.clip_sample_range)
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        variance = 0
        if t > 0:
            device = model_output.device
            variance_noise = randn_tensor(model_output.shape, generator=generator, device=device, dtype=model_output.dtype)
            if self.variance_type == "fixed_small_log":
                variance = self._get_variance(t, predicted_variance=predicted_variance) * variance_noise
            elif self.variance_type == "learned_range":
                variance = self._get_variance(t, predicted_variance=predicted_variance)
                variance = torch.exp(0.5 * variance) * variance_noise
            else:
                variance = self._get_variance(t, predicted_variance=predicted_variance)
                assert variance is not None
                variance = (variance ** 0.5) * variance_noise
        pred_prev_sample = pred_prev_sample + variance
        return {"prev_sample": pred_prev_sample, "pred_original_sample": pred_original_sample}

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, time_steps: torch.IntTensor, *, returns_velocity: bool = False) -> torch.Tensor or (torch.Tensor, torch.Tensor):
        assert self.training_mode, "The sampler is not in training mode."
        return add_noise_ddpm_ddim(self.alphas_cumprod, original_samples, noise, time_steps, returns_velocity=returns_velocity)