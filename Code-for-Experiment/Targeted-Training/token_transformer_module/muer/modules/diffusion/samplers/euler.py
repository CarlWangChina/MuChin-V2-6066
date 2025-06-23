import torch
from typing import Optional, Union, List
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.logging import get_logger
from .sampler_base import Sampler
from .consts import *
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.modules.diffusion.samplers.noise import add_noise_euler_heun
from .utils import (convert_to_karras, init_step_index, sigma_to_t, update_time_steps, torch_interp, randn_tensor)

logger = get_logger(__name__)

class EulerSampler(Sampler):
    def __init__(self, name: str, args: dict = None, training: bool = False, device: str = 'cpu'):
        super().__init__(name, args=args, training=training, device=device)
        self.prediction_type = self.args.get('prediction_type', EULER_PREDICTION_TYPE)
        self.register_buffer("sigmas", (((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5).flip(0))
        self.interpolation_type = self.args.get('interpolation_type', EULER_INTERPOLATION_TYPE)
        self.use_karras_sigmas = self.args.get('use_karras_sigmas', EULER_USE_KARRAS_SIGMAS)
        self.s_churn = self.args.get("s_churn", EULER_CHURN)
        self.s_tmin = self.args.get("s_tmin", EULER_TMIN)
        self.s_tmax = self.args.get("s_tmax", EULER_TMAX)
        self.s_noise = self.args.get("s_noise", EULER_NOISE)
        self.time_step_type = self.args.get('time_step_type', EULER_TIME_STEP_TYPE)
        if self.time_step_type == "continuous" and self.prediction_type == "v_prediction":
            self.time_steps = 0.25 * self.sigmas.log()
            self.sigmas = torch.cat([self.sigmas, torch.zeros(1, device=self.sigmas.device)])
        if not training:
            self.update_time_steps(self.num_inference_steps)

    def update_time_steps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        if self.timestep_spacing == "linspace":
            self.time_steps = torch.linspace(0, self.num_training_steps - 1, self.num_inference_steps, device=self.device).flip(0)
        else:
            self.time_steps = update_time_steps(num_training_steps=self.num_training_steps, num_inference_steps=num_inference_steps, timestep_spacing=self.timestep_spacing, steps_offset=self.steps_offset).float()
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
        log_sigmas = self.sigmas.log()
        if self.interpolation_type == "linear":
            self.sigmas = torch_interp(self.time_steps, torch.arange(0, len(self.sigmas)), self.sigmas)
        elif self.interpolation_type == "log_linear":
            self.sigmas = torch.linspace(self.sigmas[-1].log(), self.sigmas[0].log(), self.num_inference_steps + 1).exp()
        else:
            raise ValueError(f"{self.interpolation_type} is not implemented. Please specify interpolation_type to either 'linear' or 'log_linear'")
        if self.use_karras_sigmas:
            self.sigmas = convert_to_karras(in_sigmas=self.sigmas, num_inference_steps=self.num_inference_steps, sigma_min=self.sigma_min, sigma_max=self.sigma_max)
            self.time_steps = sigma_to_t(self.sigmas, log_sigmas)
        if self.time_step_type == "continuous" and self.prediction_type == "v_prediction":
            self.time_steps = 0.25 * self.sigmas.log()
        self.num_inference_steps = len(self.time_steps)
        self.sigmas = torch.cat([self.sigmas, torch.zeros(1, device=self.sigmas.device)])

    def scale_model_input(self, sample: torch.Tensor, time_step: int) -> torch.Tensor:
        step_index = init_step_index(self.time_steps, time_step)
        sigma = self.sigmas[step_index]
        sample = sample / ((sigma ** 2 + 1) ** 0.5)
        return sample

    def sample(self, model_output: torch.Tensor, time_step: int, sample: torch.Tensor, *, eta: float = 0.0, use_clipped_model_output: bool = False, generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None, variance_noise: Optional[torch.Tensor] = None, states: dict = None) -> dict:
        assert states is not None, "The EulerSampler is stateful, please provide the states store."
        if "step_index" not in states:
            states["step_index"] = init_step_index(self.time_steps, time_step)
        sigma = self.sigmas[states["step_index"]]
        gamma = min(self.s_churn / (len(self.sigmas) - 1), 2 ** 0.5 - 1) if self.s_tmin <= sigma <= self.s_tmax else 0.0
        noise = randn_tensor(model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator)
        eps = noise * self.s_noise
        sigma_hat = sigma * (gamma + 1)
        if gamma > 0:
            sample = sample + eps * (sigma_hat ** 2 - sigma ** 2) ** 0.5
        if self.prediction_type == "original_sample" or self.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.prediction_type == "epsilon":
            pred_original_sample = sample - sigma_hat * model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = model_output * (-sigma / (sigma ** 2 + 1) ** 0.5) + (sample / (sigma ** 2 + 1))
        else:
            raise ValueError(f"prediction_type given as {self.prediction_type} must be one of `epsilon`, or `v_prediction`")
        derivative = (sample - pred_original_sample) / sigma_hat
        dt = self.sigmas[states["step_index"] + 1] - sigma_hat
        prev_sample = sample + derivative * dt
        prev_sample = prev_sample.to(model_output.dtype)
        states["step_index"] += 1
        return {"prev_sample": prev_sample, "pred_original_sample": pred_original_sample}

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, time_steps: torch.IntTensor, *, returns_velocity: bool = False) -> torch.Tensor or (torch.Tensor, torch.Tensor):
        assert self.training_mode, "The sampler is not in training mode."
        assert returns_velocity is False, "The EulerSampler does not support velocity."
        schedule_time_steps = self.time_steps.to(original_samples.device)
        return add_noise_euler_heun(sigmas=self.sigmas, schedule_time_steps=schedule_time_steps, original_samples=original_samples, noise=noise, time_steps=time_steps)