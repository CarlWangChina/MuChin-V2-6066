import math
import torch
import torch.nn as nn
from music_dit.utils import get_logger
from .utils import (
    normal_kl,
    discretized_gaussian_log_likelihood,
    extract_into_tensor,
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
        posterior_mean = (
            extract_into_tensor(
                self.sampler.posterior_mean_coeff1, timesteps, samples.shape
            )
            * samples
            + extract_into_tensor(
                self.sampler.posterior_mean_coeff2, timesteps, samples.shape
            )
            * noisy_samples
        )
        posterior_log_variance = extract_into_tensor(
            self.sampler.posterior_log_variance_clipped, timesteps, samples.shape
        )
        kl_divergence = normal_kl(
            posterior_mean, posterior_log_variance, predicted_noise, predicted_log_variance
        )
        kl_divergence = mean_flat(kl_divergence) / math.log(2.0)
        decoder_nll = -discretized_gaussian_log_likelihood(
            samples, means=predicted_noise, log_scales=predicted_log_variance
        )
        assert decoder_nll.shape == samples.shape
        decoder_nll = mean_flat(decoder_nll) / math.log(2.0)
        output = torch.where(timesteps == 0, decoder_nll, kl_divergence)
        return output