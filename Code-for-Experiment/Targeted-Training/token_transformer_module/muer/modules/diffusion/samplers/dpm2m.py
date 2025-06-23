import torch
from typing import Optional, Union, List
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.logging import get_logger
import Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.modules.diffusion.samplers.noise as add_noise_dpm2m_unipc
from .sampler_base import Sampler
from .consts import *
from .utils import (convert_to_karras, convert_to_lu, threshold_sample, init_step_index, sigma_to_alpha_sigma_t, sigma_to_t, update_time_steps, torch_interp, randn_tensor)

logger = get_logger(__name__)

class DPMSolverMultistepSampler(Sampler):
    def __init__(self, name: str, args: dict = None, training: bool = False, device: str = 'cpu'):
        super(DPMSolverMultistepSampler, self).__init__(name=name, args=args, training=training, device=device)
        self.register_buffer("alpha_t", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sigma_t", torch.sqrt(1 - self.alphas_cumprod))
        self.register_buffer("lambda_t", torch.log(self.alpha_t) - torch.log(self.sigma_t))
        self.register_buffer("sigmas", ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        self.algorithm_type = self.args.get('algorithm_type', DPM2M_ALGORITHM_TYPE)
        self.solver_type = self.args.get('solver_type', DPM2M_SOLVER_TYPE)
        self.prediction_type = self.args.get('prediction_type', DPM2M_PREDICTION_TYPE)
        self.variance_type = self.args.get('variance_type', None)
        self.thresholding = self.args.get('thresholding', DPM2M_THRESHOLDING)
        self.clip_sample_range = self.args.get('clip_sample_range', DEFAULT_CLIP_SAMPLE_RANGE)
        self.dynamic_thresholding_ratio = self.args.get('dynamic_thresholding_ratio', DEFAULT_DYNAMIC_THRESHOLDING_RATIO)
        self.euler_at_final = self.args.get('euler_at_final', DPM2M_USE_EULAR_AT_FINAL)
        self.lower_order_final = self.args.get('lower_order_final', DPM2M_USE_LOWER_ORDER_FINAL)
        self.solver_order = self.args.get('solver_order', DPM2M_SOLVER_ORDER)
        self.lambda_min_clipped = self.args.get('lambda_min_clipped', DPM2M_LAMBDA_MIN_CLIPPED)
        self.use_karras_sigmas = self.args.get('use_karras_sigmas', DPM2M_USE_KARRAS_SIGMAS)
        self.use_lu_lambdas = self.args.get('use_lu_lambdas', DPM2M_USE_LU_LAMBDAS)
        self.final_sigmas_type = self.args.get('final_sigmas_type', DPM2M_FINAL_SIGMAS_TYPE)
        if self.algorithm_type not in ["dpmsolver++", "sde-dpmsolver++"] and self.final_sigmas_type == "zero":
            raise ValueError(f"`final_sigmas_type` {self.final_sigmas_type} is not supported for `algorithm_type` {self.algorithm_type}. Please choose `sigma_min` instead.")
        if not training:
            self.update_time_steps(self.num_inference_steps)

    def update_time_steps(self, num_inference_steps):
        clipped_idx = torch.searchsorted(torch.flip(self.lambda_t, [0]), self.lambda_min_clipped)
        last_timestep = (self.num_training_steps - clipped_idx).item()
        if self.timestep_spacing == "linspace":
            self.time_steps = torch.linspace(0, last_timestep - 1, num_inference_steps + 1).flip(0)[:-1].round().long()
        elif self.timestep_spacing == "leading":
            step_ratio = self.num_training_steps // (num_inference_steps + 1)
            self.time_steps = (torch.arange(0, num_inference_steps + 1) * step_ratio).flip(0)[:-1].round().long()
            self.time_steps += self.steps_offset
        elif self.timestep_spacing == "trailing":
            step_ratio = self.num_training_steps / num_inference_steps
            self.time_steps = (torch.arange(last_timestep, 0, -step_ratio)).round().long()
            self.time_steps -= 1
        else:
            raise ValueError(f"{self.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'.")
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
        log_sigmas = self.sigmas.log()
        if self.use_karras_sigmas:
            self.sigmas = self.sigmas.flip(0)
            self.sigmas = convert_to_karras(in_sigmas=self.sigmas, num_inference_steps=num_inference_steps)
            self.time_steps = sigma_to_t(self.sigmas, log_sigmas).round().long()
        elif self.use_lu_lambdas:
            lambdas = log_sigmas.flip(0)
            lambdas = convert_to_lu(in_lambdas=lambdas, num_inference_steps=num_inference_steps)
            self.sigmas = lambdas.exp().to(self.sigmas.device)
            self.time_steps = sigma_to_t(self.sigmas, log_sigmas).round().long()
        else:
            self.sigmas = torch_interp(self.time_steps, torch.arange(0, len(self.sigmas)), self.sigmas)
        if self.final_sigmas_type == "sigma_min":
            sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5
        elif self.final_sigmas_type == "zero":
            sigma_last = 0
        else:
            raise ValueError(f"`final_sigmas_type` {self.final_sigmas_type} is not supported. Please choose one of `sigma_min` or `zero`.")
        self.sigmas = torch.cat([self.sigmas, torch.Tensor([sigma_last]).to(self.sigmas.device)])
        self.num_inference_steps = len(self.time_steps)

    def _preprocess_model_output(self, model_output: torch.Tensor, sample: torch.Tensor, step_index: int) -> torch.Tensor:
        if self.algorithm_type in ["dpmsolver++", "sde-dpmsolver++"]:
            if self.prediction_type == "epsilon":
                if self.variance_type in ["learned", "learned_range"]:
                    model_output = model_output[:, :3]
                    sigma = self.sigmas[step_index]
                    alpha_t, sigma_t = sigma_to_alpha_sigma_t(sigma)
                    x0_pred = (sample - sigma_t * model_output) / alpha_t
                elif self.prediction_type == "sample":
                    x0_pred = model_output
                elif self.prediction_type == "v_prediction":
                    sigma = self.sigmas[step_index]
                    alpha_t, sigma_t = sigma_to_alpha_sigma_t(sigma)
                    x0_pred = alpha_t * sample - sigma_t * model_output
                else:
                    raise ValueError(f"prediction_type given as {self.prediction_type} must be one of `epsilon`, `sample`, or `v_prediction` for the DPMSolverMultistepScheduler.")
                if self.thresholding:
                    x0_pred = threshold_sample(x0_pred, self.dynamic_thresholding_ratio, self.clip_sample_range)
                return x0_pred
        elif self.algorithm_type in ["dpmsolver", "sde-dpmsolver"]:
            if self.prediction_type == "epsilon":
                if self.variance_type in ["learned", "learned_range"]:
                    epsilon = model_output[:, :3]
                else:
                    epsilon = model_output
            elif self.prediction_type == "sample":
                sigma = self.sigmas[step_index]
                alpha_t, sigma_t = sigma_to_alpha_sigma_t(sigma)
                epsilon = (sample - alpha_t * model_output) / sigma_t
            elif self.prediction_type == "v_prediction":
                sigma = self.sigmas[step_index]
                alpha_t, sigma_t = sigma_to_alpha_sigma_t(sigma)
                epsilon = alpha_t * model_output + sigma_t * sample
            else:
                raise ValueError(f"prediction_type given as {self.prediction_type} must be one of `epsilon`, `sample`, or `v_prediction` for the DPMSolverMultistepScheduler.")
            if self.thresholding:
                sigma = self.sigmas[step_index]
                alpha_t, sigma_t = sigma_to_alpha_sigma_t(sigma)
                x0_pred = (sample - sigma_t * epsilon) / alpha_t
                x0_pred = threshold_sample(x0_pred, self.dynamic_thresholding_ratio, self.clip_sample_range)
                epsilon = (sample - alpha_t * x0_pred) / sigma_t
            return epsilon
        else:
            raise ValueError(f"Unsupported algorithm type: {self.algorithm_type}")

    def _dpm_solver_first_order_update(self, model_output: torch.Tensor, sample: torch.Tensor, noise: torch.Tensor, step_index: int) -> torch.Tensor:
        sigma_t, sigma_s = self.sigmas[step_index + 1], self.sigmas[step_index]
        alpha_t, sigma_t = sigma_to_alpha_sigma_t(sigma_t)
        alpha_s, sigma_s = sigma_to_alpha_sigma_t(sigma_s)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s)
        h = lambda_t - lambda_s
        if self.algorithm_type == "dpmsolver++":
            return (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * model_output
        elif self.algorithm_type == "dpmsolver":
            return (alpha_t / alpha_s) * sample - (sigma_t * (torch.exp(h) - 1.0)) * model_output
        elif self.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            return (
                (sigma_t / sigma_s * torch.exp(-h)) * sample
                + (alpha_t * (1 - torch.exp(-2.0 * h))) * model_output
                + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
            )
        elif self.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            return (
                (alpha_t / alpha_s) * sample
                - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * model_output
                + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
            )
        else:
            raise ValueError(f"Unsupported algorithm type: {self.algorithm_type}")

    def _multistep_dpm_solver_second_order_update(self, model_output_list: [torch.Tensor], sample: torch.Tensor, noise: torch.Tensor, step_index: int) -> torch.Tensor:
        sigma_t, sigma_s0, sigma_s1 = (
            self.sigmas[step_index + 1],
            self.sigmas[step_index],
            self.sigmas[step_index - 1],
        )
        alpha_t, sigma_t = sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = sigma_to_alpha_sigma_t(sigma_s1)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)
        m0, m1 = model_output_list[-1], model_output_list[-2]
        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)
        if self.algorithm_type == "dpmsolver++":
            if self.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
                )
            elif self.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
                )
            else:
                raise ValueError(f"Unsupported solver type: {self.solver_type}")
        elif self.algorithm_type == "dpmsolver":
            if self.solver_type == "midpoint":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - 0.5 * (sigma_t * (torch.exp(h) - 1.0)) * D1
                )
            elif self.solver_type == "heun":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                )
            else:
                raise ValueError(f"Unsupported solver type: {self.solver_type}")
        elif self.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            if self.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                    + 0.5 * (alpha_t * (1 - torch.exp(-2.0 * h))) * D1
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
            elif self.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                    + (alpha_t * ((1.0 - torch.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * D1
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
            else:
                raise ValueError(f"Unsupported solver type: {self.solver_type}")
        elif self.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            if self.solver_type == "midpoint":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - (sigma_t * (torch.exp(h) - 1.0)) * D1
                    + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
                )
            elif self.solver_type == "heun":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - 2.0 * (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                    + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
                )
            else:
                raise ValueError(f"Unsupported solver type: {self.solver_type}")
        else:
            raise ValueError(f"Unsupported algorithm type: {self.algorithm_type}")
        return x_t

    def _multistep_dpm_solver_third_order_update(self, model_output_list: [torch.Tensor], sample: torch.Tensor, step_index: int) -> torch.Tensor:
        sigma_t, sigma_s0, sigma_s1, sigma_s2 = (
            self.sigmas[step_index + 1],
            self.sigmas[step_index],
            self.sigmas[step_index - 1],
            self.sigmas[step_index - 2],
        )
        alpha_t, sigma_t = sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = sigma_to_alpha_sigma_t(sigma_s1)
        alpha_s2, sigma_s2 = sigma_to_alpha_sigma_t(sigma_s2)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)
        lambda_s2 = torch.log(alpha_s2) - torch.log(sigma_s2)
        m0, m1, m2 = model_output_list[-1], model_output_list[-2], model_output_list[-3]
        h, h_0, h_1 = lambda_t - lambda_s0, lambda_s0 - lambda_s1, lambda_s1 - lambda_s2
        r0, r1 = h_0 / h, h_1 / h
        D0 = m0
        D1_0, D1_1 = (1.0 / r0) * (m0 - m1), (1.0 / r1) * (m1 - m2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)
        if self.algorithm_type == "dpmsolver++":
            x_t = (
                (sigma_t / sigma_s0) * sample
                - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
                - (alpha_t * ((torch.exp(-h) - 1.0 + h) / h ** 2 - 0.5)) * D2
            )
        elif self.algorithm_type == "dpmsolver":
            x_t = (
                (alpha_t / alpha_s0) * sample
                - (sigma_t * (torch.exp(h) - 1.0)) * D0
                - (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                - (sigma_t * ((torch.exp(h) - 1.0 - h) / h ** 2 - 0.5)) * D2
            )
        else:
            raise ValueError(f"Unsupported algorithm type: {self.algorithm_type} for 3rd order solver.")
        return x_t

    def sample(self, model_output: torch.Tensor, time_step: int, sample: torch.Tensor, *, eta: float = 0.0, use_clipped_model_output: bool = False, generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None, variance_noise: Optional[torch.Tensor] = None, states: dict = None) -> dict:
        assert states is not None, "The DPMSolverMultistepSampler is stateful, please provide the states store."
        if "step_index" not in states:
            states["step_index"] = init_step_index(self.time_steps, time_step)
        if "model_outputs" not in states:
            states["model_outputs"] = [None] * self.solver_order
        if "lower_order_nums" not in states:
            states["lower_order_nums"] = 0
        lower_order_final = (states["step_index"] == len(self.time_steps) - 1) and (self.euler_at_final or (self.lower_order_final and len(self.time_steps) < 15) or self.final_sigmas_type == "zero")
        lower_order_second = ((states["step_index"] == len(self.time_steps) - 2) and self.lower_order_final and len(self.time_steps) < 15)
        model_output = self._preprocess_model_output(model_output, sample=sample, step_index=states["step_index"])
        for i in range(self.solver_order - 1):
            states["model_outputs"][i] = states["model_outputs"][i + 1]
        states["model_outputs"][-1] = model_output
        if self.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"]:
            noise = randn_tensor(model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype)
        else:
            noise = None
        if self.solver_order == 1 or states["lower_order_nums"] < 1 or lower_order_final:
            prev_sample = self._dpm_solver_first_order_update(model_output, sample=sample, noise=noise, step_index=states["step_index"])
        elif self.solver_order == 2 or states["lower_order_nums"] < 2 or lower_order_second:
            prev_sample = self._multistep_dpm_solver_second_order_update(states["model_outputs"], sample=sample, noise=noise, step_index=states["step_index"])
        elif self.solver_order == 3:
            prev_sample = self._multistep_dpm_solver_third_order_update(states["model_outputs"], sample=sample, step_index=states["step_index"])
        else:
            raise ValueError(f"Unsupported solver order: {self.solver_order}")
        if states["lower_order_nums"] < self.solver_order:
            states["lower_order_nums"] += 1
        states["step_index"] += 1
        return {"prev_sample": prev_sample, "pred_original_sample": None}

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, time_steps: torch.IntTensor, *, returns_velocity: bool = False) -> torch.Tensor or (torch.Tensor, torch.Tensor):
        assert self.training_mode, "The sampler is not in training mode."
        assert returns_velocity is False, "The DPMSolverMultistepSampler does not support velocity."
        schedule_time_steps = self.time_steps.to(original_samples.device)
        return add_noise_dpm2m_unipc(sigmas=self.sigmas, schedule_time_steps=schedule_time_steps, original_samples=original_samples, noise=noise, time_steps=time_steps)