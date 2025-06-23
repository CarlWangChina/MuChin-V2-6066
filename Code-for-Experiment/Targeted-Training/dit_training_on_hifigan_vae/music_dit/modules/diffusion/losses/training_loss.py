import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from music_dit.utils import get_logger
from .vb import VariationalLowerBound
from .utils import mean_flat
from Code_for_Experiment.Targeted_Training.dit_training_on_vae.music_dit2.modules.diffusion.samplers.ddim import DDIMSampler

logger = get_logger(__name__)

class TrainingLoss(nn.Module):
    def __init__(self,
                 sampler: DDIMSampler,
                 loss_type: str = "mse"):
        super(TrainingLoss, self).__init__()
        assert loss_type in ["mse", "kl", "rescaled-kl", "rescaled-mse"]
        self.loss_type = loss_type
        self.vb = VariationalLowerBound(sampler)

    def forward(self,
                *,
                x_start: torch.Tensor,
                noise: torch.Tensor,
                noisy_samples: torch.Tensor,
                predicted_noise: torch.Tensor,
                timesteps: torch.Tensor,
                predicted_log_variance: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.loss_type == "mse" or self.loss_type == "rescaled-mse":
            return self.mse_loss(x_start=x_start,
                                 noise=noise,
                                 noisy_samples=noisy_samples,
                                 predicted_noise=predicted_noise,
                                 timesteps=timesteps,
                                 predicted_log_variance=predicted_log_variance,
                                 rescaled=(self.loss_type == "rescaled-mse")).sum()
        elif self.loss_type == "kl" or self.loss_type == "rescaled-kl":
            return self.kl_loss(x_start=x_start,
                                noisy_samples=noisy_samples,
                                predicted_noise=predicted_noise,
                                timesteps=timesteps,
                                predicted_log_variance=predicted_log_variance,
                                rescaled=(self.loss_type == "rescaled-kl")).sum()
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")

    def mse_loss(self,
                 x_start: torch.Tensor,
                 noise: torch.Tensor,
                 noisy_samples: torch.Tensor,
                 predicted_noise: torch.Tensor,
                 timesteps: torch.Tensor,
                 predicted_log_variance: Optional[torch.Tensor] = None,
                 rescaled: bool = False) -> torch.Tensor:
        mse = mean_flat((noisy_samples - x_start) ** 2)
        if predicted_log_variance is not None:
            predicted_noise_detached = predicted_noise.detach()
            vb = self.vb.get_terms_bpd(samples=x_start,
                                       noisy_samples=noisy_samples,
                                       predicted_noise=predicted_noise_detached,
                                       predicted_log_variance=predicted_log_variance,
                                       timesteps=timesteps)
            if rescaled:
                vb = vb * self.sampler.num_training_timesteps / 1000.0
            return mse + vb
        else:
            return mse

    def kl_loss(self,
                x_start: torch.Tensor,
                noisy_samples: torch.Tensor,
                predicted_noise: torch.Tensor,
                timesteps: torch.Tensor,
                predicted_log_variance: Optional[torch.Tensor],
                rescaled: bool = False) -> torch.Tensor:
        loss = self.vb.get_terms_bpd(samples=x_start,
                                     noisy_samples=noisy_samples,
                                     predicted_noise=predicted_noise,
                                     predicted_log_variance=predicted_log_variance,
                                     timesteps=timesteps)
        if rescaled:
            loss = loss * self.sampler.num_training_timesteps / 1000.0
        return loss