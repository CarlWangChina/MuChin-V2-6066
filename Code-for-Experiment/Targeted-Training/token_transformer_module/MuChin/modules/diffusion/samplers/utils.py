import math
import torch
from typing import Optional, Union, List, Tuple
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger
logger = get_logger(__name__)

def rescale_zero_terminal_snr(betas: torch.Tensor) -> torch.Tensor:
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
    alphas_bar_sqrt -= alphas_bar_sqrt_T
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
    alphas_bar = alphas_bar_sqrt**2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas
    return betas

def betas_for_alpha_bar(num_diffusion_time_steps: int, max_beta: float = 0.999, alpha_transform_type: str = "cosine"):
    if alpha_transform_type == "cosine":
        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    elif alpha_transform_type == "exp":
        def alpha_bar_fn(t):
            return math.exp(t * -12.0)
    else:
        raise ValueError(f"Unsupported alpha_tranform_type: {alpha_transform_type}")
    betas = []
    for i in range(num_diffusion_time_steps):
        t1 = i / num_diffusion_time_steps
        t2 = (i + 1) / num_diffusion_time_steps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas)

def make_beta_schedule(schedule: str, num_steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, num_steps)
    elif schedule == "scaled_linear":
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_steps) ** 2
    elif schedule == "squaredcos_cap_v2":
        return betas_for_alpha_bar(num_steps)
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, num_steps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise ValueError(f"schedule '{schedule}' is unknown.")

def update_time_steps(num_training_steps: int, num_inference_steps: int, timestep_spacing: str = "leading", steps_offset: int = 0) -> torch.Tensor:
    assert num_inference_steps <= num_training_steps, (f"The number of inference steps ({num_inference_steps}) should be less than the number of training steps ({num_training_steps}).")
    if timestep_spacing == "linspace":
        time_steps = torch.linspace(0, num_training_steps - 1, num_inference_steps).flip(0).round().long()
    elif timestep_spacing == "leading":
        step_ratio = num_training_steps // num_inference_steps
        time_steps = (torch.arange(0, num_inference_steps) * step_ratio).flip(0).round().long()
        time_steps += steps_offset
    elif timestep_spacing == "trailing":
        step_ratio = num_training_steps / num_inference_steps
        time_steps = (torch.arange(num_training_steps, 0, -step_ratio)).round().long()
        time_steps -= 1
    else:
        raise ValueError(f"{timestep_spacing} is not supported. Please make sure to choose one of 'leading' or 'trailing' or 'linspace'.")
    assert time_steps.shape == (num_inference_steps,)
    return time_steps

def threshold_sample(sample: torch.Tensor, dynamic_thresholding_ratio: float, sample_max_value: float) -> torch.Tensor:
    dtype = sample.dtype
    batch_size, channels, *remaining_dims = sample.shape
    if dtype not in (torch.float32, torch.float64):
        sample = sample.float()
    new_channels = channels * torch.prod(torch.tensor(remaining_dims)).long().item()
    sample = sample.reshape(batch_size, new_channels)
    abs_sample = sample.abs()
    s = torch.quantile(abs_sample, dynamic_thresholding_ratio, dim=1)
    s = torch.clamp(s, min=1, max=sample_max_value)
    s = s.unsqueeze(1)
    sample = torch.clamp(sample, -s, s) / s
    sample = sample.reshape(batch_size, channels, *remaining_dims)
    sample = sample.to(dtype)
    return sample

def convert_to_karras(in_sigmas: torch.Tensor, num_inference_steps: int, sigma_min: Optional[float] = None, sigma_max: Optional[float] = None) -> torch.Tensor:
    sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
    sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()
    rho = 7.0
    ramp = torch.linspace(0, 1, num_inference_steps)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas.to(in_sigmas.device)

def convert_to_lu(in_lambdas: torch.FloatTensor, num_inference_steps) -> torch.FloatTensor:
    lambda_min = in_lambdas[-1].item()
    lambda_max = in_lambdas[0].item()
    rho = 1.0
    ramp = torch.linspace(0, 1, num_inference_steps)
    min_inv_rho = lambda_min ** (1 / rho)
    max_inv_rho = lambda_max ** (1 / rho)
    lambdas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return lambdas

def init_step_index(time_steps: torch.Tensor, time_step: int) -> torch.Tensor:
    if isinstance(time_step, torch.Tensor):
        time_step = time_step.to(time_steps.device)
    index_candidates = (time_steps == time_step).nonzero()
    if len(index_candidates) == 0:
        step_index = len(time_steps) - 1
    elif len(index_candidates) > 1:
        step_index = index_candidates[1].item()
    else:
        step_index = index_candidates[0].item()
    return step_index

def sigma_to_alpha_sigma_t(sigma):
    alpha_t = 1 / ((sigma ** 2 + 1) ** 0.5)
    sigma_t = sigma * alpha_t
    return alpha_t, sigma_t

def sigma_to_t(sigma, log_sigmas):
    log_sigma = sigma.clamp(min=1e-4).log()
    dists = log_sigma - log_sigmas.unsqueeze(-1)
    low_idx = torch.cumsum((dists >= 0), dim=0).argmax(dim=0).clamp(max=log_sigmas.shape[0] - 2)
    high_idx = low_idx + 1
    low = log_sigmas[low_idx]
    high = log_sigmas[high_idx]
    w = (low - log_sigma) / (low - high)
    w = w.clamp(min=0, max=1)
    t = (1 - w) * low_idx + w * high_idx
    t = t.reshape(sigma.shape)
    return t

def torch_interp(x, xp, fp):
    idxs = torch.argsort(xp)
    xp_sorted = xp[idxs].to(x.device)
    fp_sorted = fp[idxs].to(x.device)
    idx = torch.searchsorted(xp_sorted, x, right=True)
    idx = torch.clamp(idx, 1, len(xp_sorted) - 1)
    xp_low = xp_sorted[idx - 1]
    xp_high = xp_sorted[idx]
    fp_low = fp_sorted[idx - 1]
    fp_high = fp_sorted[idx]
    slope = (fp_high - fp_low) / (xp_high - xp_low)
    interp_values = fp_low + slope * (x - xp_low)
    return interp_values

def randn_tensor(shape: Union[Tuple, List], generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None, device: Optional["torch.device"] = None, dtype: Optional["torch.dtype"] = None, layout: Optional["torch.layout"] = None):
    rand_device = device
    batch_size = shape[0]
    layout = layout or torch.strided
    device = device or torch.device("cpu")
    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                logger.debug(f"The passed generator was created on 'cpu' even though a tensor on {device} was expected. Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably slightly speed up this function by passing a generator that was created on the {device} device.")
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]
        if isinstance(generator, list):
            shape = (1,) + shape[1:]
            latents = [torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout) for i in range(batch_size)]
            latents = torch.cat(latents, dim=0).to(device)
        else:
            latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)
    return latents