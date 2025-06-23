import torch
from typing import Optional, List, Union
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.logging import get_logger
import Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.modules.diffusion.samplers.noise as add_noise_dpm2m_unipc
from .sampler_base import Sampler
from .consts import *
from .utils import (convert_to_karras, init_step_index, threshold_sample, sigma_to_alpha_sigma_t, sigma_to_t, torch_interp)

logger = get_logger(__name__)

class UniPCMultistepSampler(Sampler):
    def __init__(self, name: str, args: dict = None, training: bool = False, device: str = 'cpu'):
        super().__init__(name, args, training=training, device=device)
        self.alpha_t = torch.sqrt(self.alphas_cumprod)
        self.sigma_t = torch.sqrt(1 - self.alphas_cumprod)
        self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)
        self.use_karras_sigmas = self.args.get('use_karras_sigmas', UNIPC_USE_KARRAS_SIGMAS)
        self.register_buffer("sigmas", (((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5))
        self.solver_type = self.args.get('solver_type', UNIPC_SOLVER_TYPE)
        self.prediction_type = self.args.get('prediction_type', UNIPC_PREDICTION_TYPE)
        self.predict_x0 = self.args.get('predict_x0', UNIPC_PREDICT_X0)
        self.thresholding = self.args.get('thresholding', UNIPC_THRESHOLDING)
        self.clip_sample_range = self.args.get('clip_sample_range', DEFAULT_CLIP_SAMPLE_RANGE)
        self.dynamic_thresholding_ratio = self.args.get('dynamic_thresholding_ratio', DEFAULT_DYNAMIC_THRESHOLDING_RATIO)
        self.solver_order = self.args.get('solver_order', UNIPC_SOLVER_ORDER)
        self.lower_order_final = self.args.get('lower_order_final', UNIPC_USE_LOWER_ORDER_FINAL)
        self.disable_corrector = self.args.get('disable_corrector', UNIPC_DISABLE_CORRECTOR)
        if not training:
            self.update_time_steps(self.num_inference_steps)

    def update_time_steps(self, num_inference_steps):
        if self.timestep_spacing == "linspace":
            self.time_steps = torch.linspace(0, self.num_training_steps - 1, num_inference_steps + 1).flip(0)[:-1].round().long()
        elif self.timestep_spacing == "leading":
            step_ratio = self.num_training_steps // (num_inference_steps + 1)
            self.time_steps = (torch.arange(0, num_inference_steps + 1) * step_ratio).flip(0)[:-1].round().long()
            self.time_steps += self.steps_offset
        elif self.timestep_spacing == "trailing":
            step_ratio = self.num_training_steps / num_inference_steps
            self.time_steps = (torch.arange(self.num_training_steps, 0, -step_ratio)).round().long()
            self.time_steps -= 1
        else:
            raise ValueError(f"{self.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'.")
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
        if self.use_karras_sigmas:
            log_sigmas = self.sigmas.log()
            self.sigmas = self.sigmas.flip(0)
            self.sigmas = convert_to_karras(in_sigmas=self.sigmas, num_inference_steps=self.num_inference_steps)
            self.time_steps = sigma_to_t(self.sigmas, log_sigmas).round().long()
            self.sigmas = torch.cat([self.sigmas, self.sigmas.flip(0)])
        else:
            self.sigmas = torch_interp(self.time_steps, torch.arange(0, len(self.sigmas)), self.sigmas)
            sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5
            self.sigmas = torch.cat([self.sigmas, torch.Tensor([sigma_last])])
        self.num_inference_steps = len(self.time_steps)

    def _convert_model_output(self, model_output: torch.Tensor, step_index: int, sample: torch.Tensor = None) -> torch.Tensor:
        sigma = self.sigmas[step_index]
        alpha_t, sigma_t = sigma_to_alpha_sigma_t(sigma)
        if self.predict_x0:
            if self.prediction_type == "epsilon":
                x0_pred = (sample - sigma_t * model_output) / alpha_t
            elif self.prediction_type == "sample":
                x0_pred = model_output
            elif self.prediction_type == "v_prediction":
                x0_pred = alpha_t * sample - sigma_t * model_output
            else:
                raise ValueError(f"prediction_type given as {self.prediction_type} must be one of `epsilon`, `sample`, or `v_prediction` for the UniPCMultistepScheduler.")
            if self.thresholding:
                x0_pred = threshold_sample(x0_pred, self.dynamic_thresholding_ratio, self.clip_sample_range)
            return x0_pred
        else:
            if self.prediction_type == "epsilon":
                return model_output
            elif self.prediction_type == "sample":
                epsilon = (sample - alpha_t * model_output) / sigma_t
                return epsilon
            elif self.prediction_type == "v_prediction":
                epsilon = alpha_t * model_output + sigma_t * sample
                return epsilon
            else:
                raise ValueError(f"prediction_type given as {self.prediction_type} must be one of `epsilon`, `sample`, or `v_prediction` for the UniPCMultistepScheduler.")

    def _multistep_uni_p_bh_update(self, model_output_list: List[torch.Tensor], step_index: int, sample: torch.Tensor, order: int = None) -> torch.Tensor:
        m0 = model_output_list[-1]
        x = sample
        sigma_t, sigma_s0 = self.sigmas[step_index + 1], self.sigmas[step_index]
        alpha_t, sigma_t = sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = sigma_to_alpha_sigma_t(sigma_s0)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0
        device = sample.device
        rks = []
        D1s = []
        for i in range(1, order):
            si = step_index - i
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)
        rks.append(1.0)
        rks = torch.tensor(rks, device=device)
        R = []
        b = []
        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1
        if self.solver_type == "bh1":
            B_h = hh
        elif self.solver_type == "bh2":
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError(f"Solver type {self.solver_type} is not implemented.")
        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i
        R = torch.stack(R)
        b = torch.tensor(b, device=device)
        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
            if order == 2:
                rhos_p = torch.tensor([0.5], dtype=x.dtype, device=device)
            else:
                rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1]).to(dtype=x.dtype, device=device)
        else:
            D1s = None
            rhos_p = None
        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if D1s is not None:
                assert rhos_p is not None
                pred_res = torch.einsum("k, bkc... -> bc...", rhos_p, D1s)
            else:
                pred_res = 0
            x_t = x_t_ - alpha_t * B_h * pred_res
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            if D1s is not None:
                assert rhos_p is not None
                pred_res = torch.einsum("k, bkc... -> bc...", rhos_p, D1s)
            else:
                pred_res = 0
            x_t = x_t_ - sigma_t * B_h * pred_res
        x_t = x_t.to(x.dtype)
        return x_t

    def _multistep_uni_c_bh_update(self, this_model_output: torch.Tensor, model_output_list: List[torch.Tensor], step_index: int, last_sample: torch.Tensor, this_sample: torch.Tensor, order: int = None) -> torch.Tensor:
        m0 = model_output_list[-1]
        x = last_sample
        model_t = this_model_output
        sigma_t, sigma_s0 = self.sigmas[step_index], self.sigmas[step_index - 1]
        alpha_t, sigma_t = sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = sigma_to_alpha_sigma_t(sigma_s0)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0
        device = this_sample.device
        rks = []
        D1s = []
        for i in range(1, order):
            si = step_index - (i + 1)
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)
        rks.append(1.0)
        rks = torch.tensor(rks, device=device)
        R = []
        b = []
        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1
        if self.solver_type == "bh1":
            B_h = hh
        elif self.solver_type == "bh2":
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()
        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i
        R = torch.stack(R)
        b = torch.tensor(b, device=device)
        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
        else:
            D1s = None
        if order == 1:
            rhos_c = torch.tensor([0.5], dtype=x.dtype, device=device)
        else:
            rhos_c = torch.linalg.solve(R, b).to(dtype=x.dtype, device=device)
        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if D1s is not None:
                corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
            else:
                corr_res = 0
            D1_t = model_t - m0
            x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            if D1s is not None:
                corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
            else:
                corr_res = 0
            D1_t = model_t - m0
            x_t = x_t_ - sigma_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        x_t = x_t.to(x.dtype)
        return x_t

    def sample(self, model_output: torch.Tensor, time_step: int, sample: torch.Tensor, eta: float = 0.0, use_clipped_model_output: bool = False, generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None, variance_noise: Optional[torch.Tensor] = None, states: dict = None) -> dict:
        assert states is not None, "The UniPCMultistepSampler is stateful, please provide the states store."
        if "step_index" not in states:
            states["step_index"] = init_step_index(self.time_steps, time_step)
        if "model_outputs" not in states:
            states["model_outputs"] = [None] * self.solver_order
        if "lower_order_nums" not in states:
            states["lower_order_nums"] = 0
        if "last_sample" not in states:
            states["last_sample"] = None
        if "current_order" not in states:
            states["current_order"] = None
        use_corrector = (states["step_index"] > 0 and states["step_index"] - 1 not in self.disable_corrector and states["last_sample"] is not None)
        model_output_converted = self._convert_model_output(model_output, step_index=states["step_index"], sample=sample)
        if use_corrector:
            sample = self._multistep_uni_c_bh_update(this_model_output=model_output_converted, model_output_list=states["model_outputs"], step_index=states["step_index"], last_sample=states["last_sample"], this_sample=sample, order=states["current_order"])
        for i in range(self.solver_order - 1):
            states["model_outputs"][i] = states["model_outputs"][i + 1]
        states["model_outputs"][-1] = model_output_converted
        if self.lower_order_final:
            current_order = min(self.solver_order, len(self.time_steps) - states["step_index"])
        else:
            current_order = self.solver_order
        states["current_order"] = min(current_order, states["lower_order_nums"] + 1)
        assert states["current_order"] > 0
        states["last_sample"] = sample
        prev_sample = self._multistep_uni_p_bh_update(model_output_list=states["model_outputs"], sample=sample, step_index=states["step_index"], order=states["current_order"])
        if states["lower_order_nums"] < self.solver_order:
            states["lower_order_nums"] += 1
        states["step_index"] += 1
        return {"prev_sample": prev_sample, "pred_original_sample": None}

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, time_steps: torch.IntTensor, returns_velocity: bool = False) -> torch.Tensor or (torch.Tensor, torch.Tensor):
        assert self.training_mode, "The sampler is not in training mode."
        assert returns_velocity is False, "The UniPCMultistepSampler does not support velocity."
        schedule_time_steps = self.time_steps.to(original_samples.device)
        return add_noise_dpm2m_unipc(sigmas=self.sigmas, schedule_time_steps=schedule_time_steps, original_samples=original_samples, noise=noise, time_steps=time_steps)