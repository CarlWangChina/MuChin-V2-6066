import math
import torch
import torch.nn as nn
from ama_prof_divi_common.utils import get_logger
from .utils import (
    normal_kl,
    discretized_gaussian_log_likelihood,
    mean_flat,
)
from Code_for_Experiment.Targeted_Training.dit_training_on_vae.music_dit2.modules.diffusion.samplers.ddim import DDIMSampler

logger = get_logger(__name__)


class VariationalLowerBound(nn.Module):
    def __init__(self, sampler: DDIMSampler):
        super(VariationalLowerBound, self).__init__()
        self.sampler = sampler

    def get_terms_bpd(
        self,
        samples: torch.Tensor,
        noisy_samples: torch.Tensor,
        predicted_noise: torch.Tensor,
        predicted_log_variance: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        posterior_mean, posterior_log_variance = self.sampler.q_posterior_mean_variance(
            samples=samples, noisy_samples=noisy_samples, timesteps=timesteps
        )
        prev_predicted_mean, _, _, prev_predicted_log_variance = self.sampler.denoise(
            model_output=(predicted_noise, predicted_log_variance),
            timesteps=timesteps,
            sample=samples,
        )
        kl_divergence = normal_kl(
            posterior_mean,
            posterior_log_variance,
            prev_predicted_mean,
            prev_predicted_log_variance,
        )
        kl_divergence = mean_flat(kl_divergence) / math.log(2.0)
        decoder_nll = -discretized_gaussian_log_likelihood(
            samples, means=prev_predicted_mean, log_scales=prev_predicted_log_variance * 0.5
        )
        decoder_nll = mean_flat(decoder_nll) / math.log(2.0)
        output = torch.where(timesteps == 0, decoder_nll, kl_divergence)
        return output