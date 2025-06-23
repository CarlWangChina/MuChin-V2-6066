import torch
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.logging import get_logger
import sigma_to_alpha_sigma
_tlogger = get_logger(__name__)

def add_noise_ddpm_ddim(alphas_cumprod: torch.Tensor, original_samples: torch.Tensor, noise: torch.Tensor, time_steps: torch.IntTensor, returns_velocity: bool = False) -> torch.Tensor or (torch.Tensor, torch.Tensor):
    alphas_cumprod = alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
    time_steps = time_steps.round().long().to(original_samples.device)
    sqrt_alpha_prod = alphas_cumprod[time_steps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[time_steps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
    noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    if returns_velocity:
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * original_samples
        return noisy_samples, velocity
    else:
        return noisy_samples

def add_noise_euler_heun(sigmas: torch.Tensor, schedule_time_steps: torch.IntTensor, original_samples: torch.Tensor, noise: torch.Tensor, time_steps: torch.IntTensor) -> torch.Tensor:
    sigmas = sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
    if original_samples.device.type == "mps" and torch.is_floating_point(time_steps):
        schedule_time_steps = schedule_time_steps.to(dtype=torch.float32)
        time_steps = time_steps.to(original_samples.device, dtype=torch.float32)
    else:
        time_steps = time_steps.to(original_samples.device)
    step_indices = []
    for time_step in time_steps:
        index_candidates = (schedule_time_steps == time_step).nonzero()
        if len(index_candidates) == 0:
            step_index = len(schedule_time_steps) - 1
        elif len(index_candidates) > 1:
            step_index = index_candidates[1].item()
        else:
            step_index = index_candidates[0].item()
        step_indices.append(step_index)
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < len(original_samples.shape):
        sigma = sigma.unsqueeze(-1)
    noisy_samples = original_samples + noise * sigma
    return noisy_samples

def add_noise_dpm2m_unipc(sigmas: torch.Tensor, schedule_time_steps: torch.IntTensor, original_samples: torch.Tensor, noise: torch.Tensor, time_steps: torch.IntTensor) -> torch.Tensor:
    sigmas = sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
    if original_samples.device.type == "mps" and torch.is_floating_point(time_steps):
        schedule_time_steps = schedule_time_steps.to(dtype=torch.float32)
        time_steps = time_steps.to(original_samples.device, dtype=torch.float32)
    else:
        time_steps = time_steps.to(original_samples.device)
    step_indices = []
    for time_step in time_steps:
        index_candidates = (schedule_time_steps == time_step).nonzero()
        if len(index_candidates) == 0:
            step_index = len(schedule_time_steps) - 1
        elif len(index_candidates) > 1:
            step_index = index_candidates[1].item()
        else:
            step_index = index_candidates[0].item()
        step_indices.append(step_index)
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < len(original_samples.shape):
        sigma = sigma.unsqueeze(-1)
    alpha_t, sigma_t = sigma_to_alpha_sigma(sigma)
    noisy_samples = alpha_t * original_samples + sigma_t * noise
    return noisy_samples