import torch
import math

from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger
logger = get_logger(__name__)

def betas_for_alpha_bar(num_diffusion_time_steps: int, max_beta: float = 0.999, alpha_transform_type: str = "cosine"):
    if alpha_transform_type == "cosine":
        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    elif alpha_transform_type == "exp":
        def alpha_bar_fn(t):
            return math.exp(t * -12.0)
    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_time_steps):
        t1 = i / num_diffusion_time_steps
        t2 = (i + 1) / num_diffusion_time_steps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)

def make_beta_schedule(schedule: str, num_steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
    elif schedule == "scaled_linear":
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_steps, dtype=torch.float32) ** 2
    elif schedule == "squaredcos_cap_v2":
        return betas_for_alpha_bar(num_steps)
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, num_steps, dtype=torch.float32)
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

def threshold_sample(sample: torch.Tensor, *, dynamic_thresholding_ratio: float) -> torch.Tensor:
    dtype = sample.dtype
    batch_size, channels, *remaining_dims = sample.shape
    if dtype not in (torch.float32, torch.float64):
        sample = sample.float()
    sample = sample.reshape(batch_size, channels * torch.prod(torch.Tensor(remaining_dims)).long().item())
    abs_sample = sample.abs()
    s = torch.quantile(abs_sample, dynamic_thresholding_ratio, dim=1)
    s = s.clamp(-1.0, 1.0)
    s = s.unsqueeze(1)
    sample = torch.clamp(sample, -s, s) / s
    sample = sample.reshape(batch_size, channels, *remaining_dims)
    sample = sample.to(dtype)
    return sample

def extract_into_tensor(arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: torch.Size) -> torch.Tensor:
    res = arr[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape).to(res.device)