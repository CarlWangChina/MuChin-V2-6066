import torch
from typing import Optional

from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.logging import get_logger
from .sampler_base import Sampler
from .consts import *
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.modules.diffusion.samplers.noise import add_noise_euler_heun
from .utils import convert_to_karras, init_step_index, sigma_to_t, update_time_steps, torch_interp

logger = get_logger(__name__)

class HeunSampler(Sampler):
    def __init__(self, name: str, args: dict = None, training: bool = False, device: str = 'cpu'):
        super().__init__(name, args, training=training, device=device)
        self.prediction_type = self.args.get('prediction_type', HEUN_PREDICTION_TYPE)
        self.use_karras_sigmas = self.args.get('use_karras_sigmas', HEUN_USE_KARRAS_SIGMAS)
        sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
        self.register_buffer("sigmas", sigmas)
        self.clip_sample = self.args.get('clip_sample', HEUN_CLIP_SAMPLE)
        self.clip_sample_range = self.args.get('clip_sample_range', DEFAULT_CLIP_SAMPLE_RANGE)
        self.gamma = self.args.get('gamma', HEUN_GAMMA)
        if training:
            self.update_time_steps(self.num_training_steps)
        else:
            self.update_time_steps(self.num_inference_steps)

    def update_time_steps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        if self.timestep_spacing == "linspace":
            self.time_steps = torch.linspace(0, self.num_training_steps - 1, self.num_inference_steps, device=self.device).flip(0)
        else:
            self.time_steps = update_time_steps(num_training_steps=self.num_training_steps, num_inference_steps=num_inference_steps, timestep_spacing=self.timestep_spacing, steps_offset=self.steps_offset).float()
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
        log_sigmas = self.sigmas.log()
        self.sigmas = torch_interp(self.time_steps, torch.arange(0, len(self.sigmas)), self.sigmas)
        if self.use_karras_sigmas:
            self.sigmas = convert_to_karras(in_sigmas=self.sigmas, num_inference_steps=self.num_inference_steps, sigma_min=self.sigma_min, sigma_max=self.sigma_max).to(self.device)
        self.time_steps = sigma_to_t(self.sigmas, log_sigmas)
        self.sigmas = torch.cat([self.sigmas, torch.Tensor([0.0]).to(self.sigmas.device)])
        self.sigmas = torch.cat([self.sigmas[:1], self.sigmas[1:-1].repeat_interleave(2), self.sigmas[-1:]])
        self.time_steps = torch.cat([self.time_steps[:1], self.time_steps[1:].repeat_interleave(2)])
        self.num_inference_steps = len(self.time_steps)

    def scale_model_input(self, sample: torch.Tensor, time_step: int) -> torch.Tensor:
        step_index = init_step_index(self.time_steps, time_step)
        sigma = self.sigmas[step_index]
        sample = sample / ((sigma ** 2 + 1) ** 0.5)
        return sample

    def sample(self, model_output: torch.Tensor, time_step: int, sample: torch.Tensor, *, eta: float = 0.0, use_clipped_model_output: bool = False, generator: torch.Generator = None, variance_noise: Optional[torch.Tensor] = None, states: dict = None) -> dict:
        assert states is not None, "The HeunSampler is stateful, please provide the states store."
        if "step_index" not in states:
            states["step_index"] = init_step_index(self.time_steps, time_step)
        if "dt" not in states:
            states["dt"] = None
        if states["dt"] is None:
            sigma = self.sigmas[states["step_index"]]
            sigma_next = self.sigmas[states["step_index"] + 1]
        else:
            sigma = self.sigmas[states["step_index"] - 1]
            sigma_next = self.sigmas[states["step_index"]]
        sigma_hat = sigma * (self.gamma + 1)
        if self.prediction_type == "epsilon":
            sigma_input = sigma_hat if states["dt"] is None else sigma_next
            pred_original_sample = sample - sigma_input * model_output
        elif self.prediction_type == "v_prediction":
            sigma_input = sigma_hat if states["dt"] is None else sigma_next
            pred_original_sample = model_output * (-sigma_input / (sigma_input ** 2 + 1) ** 0.5) + (sample / (sigma_input ** 2 + 1))
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        else:
            raise ValueError(f"prediction_type given as {self.prediction_type} must be one of `epsilon`, `simple`, or `v_prediction`")
        if self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-self.clip_sample_range, self.clip_sample_range)
        if states["dt"] is None:
            derivative = (sample - pred_original_sample) / sigma_hat
            dt = sigma_next - sigma_hat
            states["prev_derivative"] = derivative
            states["dt"] = dt
            states["sample"] = sample
        else:
            derivative = (sample - pred_original_sample) / sigma_next
            derivative = (states["prev_derivative"] + derivative) / 2
            dt = states["dt"]
            sample = states["sample"]
            states["prev_derivative"] = None
            states["dt"] = None
            states["sample"] = None
        prev_sample = sample + derivative * dt
        states["step_index"] += 1
        return {"prev_sample": prev_sample, "pred_original_sample": pred_original_sample}

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, time_steps: torch.IntTensor, returns_velocity: bool = False) -> torch.Tensor or (torch.Tensor, torch.Tensor):
        assert self.training_mode, "The sampler is not in training mode."
        assert returns_velocity is False, "The HeunSampler does not support velocity."
        schedule_time_steps = self.time_steps.to(original_samples.device)
        return add_noise_euler_heun(sigmas=self.sigmas, schedule_time_steps=schedule_time_steps, original_samples=original_samples, noise=noise, time_steps=time_steps)