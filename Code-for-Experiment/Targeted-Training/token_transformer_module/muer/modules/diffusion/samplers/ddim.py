import torch
from typing import Optional, Union, List
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.logging import get_logger
logger = get_logger(__name__)
from .sampler_base import Sampler
from .utils import threshold_sample, randn_tensor
from .consts import *

class DDIMSampler(Sampler):
    def __init__(self, name: str, args: dict = None, training: bool = False, device: str = 'cpu'):
        super(DDIMSampler, self).__init__(name=name, args=args, training=training, device=device)
        set_alpha_to_one = self.args.get('set_alpha_to_one', DDIM_SET_ALPHA_TO_ONE)
        self.register_buffer("final_alpha_cumprod", torch.tensor([1.0], device=device) if set_alpha_to_one else self.alphas_cumprod[0])
        self.prediction_type = self.args.get('prediction_type', DDIM_PREDICTION_TYPE)
        self.thresholding = self.args.get('thresholding', DDIM_THRESHOLDING)
        self.clip_sample = self.args.get('clip_sample', DDIM_CLIP_SAMPLE)
        self.clip_sample_range = self.args.get('clip_sample_range', DEFAULT_CLIP_SAMPLE_RANGE)
        self.dynamic_thresholding_ratio = self.args.get('dynamic_thresholding_ratio', DEFAULT_DYNAMIC_THRESHOLDING_RATIO)
        if not training:
            self.update_time_steps(self.num_inference_steps)

    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance

    def sample(self, model_output: torch.Tensor, time_step: int, sample: torch.Tensor, *, eta: float = 0.0, use_clipped_model_output: bool = False, generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None, variance_noise: Optional[torch.Tensor] = None, states: dict = None) -> dict:
        step_ratio = self.num_training_steps // self.num_inference_steps
        prev_time_step = time_step - step_ratio
        alpha_prod_t = self.alphas_cumprod[time_step]
        alpha_prod_t_prev = self.alphas_cumprod[prev_time_step] if prev_time_step >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
            pred_epsilon = model_output
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
            pred_epsilon = (alpha_prod_t ** 0.5) * model_output + (beta_prod_t ** 0.5) * sample
        else:
            raise ValueError(f"prediction_type given as {self.prediction_type} must be one of `epsilon`, `sample`, or `v_prediction`.")
        if self.thresholding:
            pred_original_sample = threshold_sample(pred_original_sample, self.dynamic_thresholding_ratio, self.clip_sample_range)
        elif self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -self.clip_sample_range, self.clip_sample_range)
        variance = self._get_variance(time_step, prev_time_step)
        std_dev_t = eta * variance ** 0.5
        if use_clipped_model_output:
            pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** 0.5 * pred_epsilon
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        if eta > 0:
            if variance_noise is not None and generator is not None:
                raise ValueError("Cannot pass both generator and variance_noise. Please make sure that either `generator` or `variance_noise` stays `None`.")
            if variance_noise is None:
                variance_noise = randn_tensor(model_output.shape, generator=generator, device=self.device, dtype=model_output.dtype)
            variance = std_dev_t * variance_noise
            prev_sample = prev_sample + variance
        return {"prev_sample": prev_sample, "pred_original_sample": pred_original_sample}

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, time_steps: torch.IntTensor, *, returns_velocity: bool = False) -> torch.Tensor or (torch.Tensor, torch.Tensor):
        assert self.training_mode, "The sampler is not in training mode."
        return add_noise_ddpm_ddim(self.alphas_cumprod, original_samples, noise, time_steps, returns_velocity)