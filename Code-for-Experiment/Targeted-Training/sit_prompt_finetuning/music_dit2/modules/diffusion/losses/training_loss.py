import torch
import torch.nn as nn
from typing import Optional, Dict
from ama_prof_divi_common.utils import get_logger
from .vlb import VariationalLowerBound
from .utils import mean_flat
from Code_for_Experiment.Targeted_Training.dit_training_on_vae.music_dit2.modules.diffusion.samplers.ddim import DDIMSampler

logger = get_logger(__name__)

class TrainingLoss(nn.Module):
    def __init__(self,
                 sampler: DDIMSampler,
                 loss_type: str = "mse",
                 mse_loss_weight: float = 1.0,
                 vb_loss_weight: float = 1.0):
        super(TrainingLoss, self).__init__()
        assert loss_type in ["mse", "kl", "rescaled-kl", "rescaled-mse"]
        self.loss_type = loss_type
        self.mse_loss_weight = mse_loss_weight
        self.vb_loss_weight = vb_loss_weight
        self.sampler = sampler
        self.vb = VariationalLowerBound(sampler)

    def forward(self,
                *,
                x_start: torch.Tensor,
                noise: torch.Tensor,
                noisy_samples: torch.Tensor,
                timesteps: torch.Tensor,
                predicted_noise: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                predicted_log_variance: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if mask is not None:
            mask = mask.unsqueeze(-1)
            x_start = x_start * mask
            noise = noise * mask
            noisy_samples = noisy_samples * mask
            predicted_noise = predicted_noise * mask
            if predicted_log_variance is not None:
                predicted_log_variance = predicted_log_variance * mask

        if self.loss_type == "mse" or self.loss_type == "rescaled-mse":
            return self.mse_loss(x_start=x_start,
                                 noise=noise,
                                 noisy_samples=noisy_samples,
                                 predicted_noise=predicted_noise,
                                 timesteps=timesteps,
                                 predicted_log_variance=predicted_log_variance,
                                 rescaled=(self.loss_type == "rescaled-mse"))
        elif self.loss_type == "kl" or self.loss_type == "rescaled-kl":
            return self.kl_loss(x_start=x_start,
                                noisy_samples=noisy_samples,
                                predicted_noise=predicted_noise,
                                timesteps=timesteps,
                                predicted_log_variance=predicted_log_variance,
                                rescaled=(self.loss_type == "rescaled-kl"))
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")

    def mse_loss(self,
                 x_start: torch.Tensor,
                 noise: torch.Tensor,
                 noisy_samples: torch.Tensor,
                 predicted_noise: torch.Tensor,
                 timesteps: torch.Tensor,
                 predicted_log_variance: Optional[torch.Tensor] = None,
                 rescaled: bool = False) -> Dict[str, torch.Tensor]:
        mse = mean_flat((predicted_noise - noise) ** 2) * self.mse_loss_weight
        if predicted_log_variance is not None:
            predicted_noise_detached = predicted_noise.detach()
            vb = self.vb.get_terms_bpd(samples=x_start,
                                       noisy_samples=noisy_samples,
                                       predicted_noise=predicted_noise_detached,
                                       predicted_log_variance=predicted_log_variance,
                                       timesteps=timesteps) * self.vb_loss_weight
            if rescaled:
                vb = vb * self.sampler.num_training_timesteps / 1000.0
            mse = mse.mean()
            vb = vb.mean()
            return {
                "x_start_min": x_start.min(),
                "x_start_max": x_start.max(),
                "x_start_mean": x_start.mean(),
                "x_start_std": x_start.std(),
                "noise_min": noise.min(),
                "noise_max": noise.max(),
                "noise_mean": noise.mean(),
                "noise_std": noise.std(),
                "predicted_noise_min": predicted_noise.min(),
                "predicted_noise_max": predicted_noise.max(),
                "predicted_noise_mean": predicted_noise.mean(),
                "predicted_noise_std": predicted_noise.std(),
                "predicted_log_variance_min": predicted_log_variance.min(),
                "predicted_log_variance_max": predicted_log_variance.max(),
                "predicted_log_variance_mean": predicted_log_variance.mean(),
                "predicted_log_variance_std": predicted_log_variance.std(),
                "mse_loss": mse,
                "vb_loss": vb,
                "total_loss": mse + vb
            }
        else:
            mse = mse.mean()
            return {
                "x_start_min": x_start.min(),
                "x_start_max": x_start.max(),
                "x_start_mean": x_start.mean(),
                "x_start_std": x_start.std(),
                "noise_min": noise.min(),
                "noise_max": noise.max(),
                "noise_mean": noise.mean(),
                "noise_std": noise.std(),
                "predicted_noise_min": predicted_noise.min(),
                "predicted_noise_max": predicted_noise.max(),
                "predicted_noise_mean": predicted_noise.mean(),
                "predicted_noise_std": predicted_noise.std(),
                "mse_loss": mse,
                "vb_loss": torch.tensor(0.0).to(mse.device),
                "total_loss": mse
            }

    def kl_loss(self,
                x_start: torch.Tensor,
                noisy_samples: torch.Tensor,
                predicted_noise: torch.Tensor,
                timesteps: torch.Tensor,
                predicted_log_variance: Optional[torch.Tensor],
                rescaled: bool = False) -> Dict[str, torch.Tensor]:
        loss = self.vb.get_terms_bpd(samples=x_start,
                                     noisy_samples=noisy_samples,
                                     predicted_noise=predicted_noise,
                                     predicted_log_variance=predicted_log_variance,
                                     timesteps=timesteps).sum()
        if rescaled:
            loss = loss * self.sampler.num_training_timesteps / 1000.0
        loss = loss.mean()
        return {
            "x_start_min": x_start.min(),
            "x_start_max": x_start.max(),
            "x_start_mean": x_start.mean(),
            "x_start_std": x_start.std(),
            "predicted_noise_min": predicted_noise.min(),
            "predicted_noise_max": predicted_noise.max(),
            "predicted_noise_mean": predicted_noise.mean(),
            "predicted_noise_std": predicted_noise.std(),
            "kl_loss": loss,
            "total_loss": loss
        }