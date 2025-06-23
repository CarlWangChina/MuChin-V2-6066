from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Optional, Union, List
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger
from .utils import make_beta_schedule, update_time_steps, rescale_zero_terminal_snr
from .consts import *

logger = get_logger(__name__)

class Sampler(ABC, nn.Module):
    def __init__(self, 
                 name: str, 
                 args: dict = None, 
                 *, 
                 training: bool = False, 
                 device: Union[torch.device, str] = "cpu"):
        super(Sampler, self).__init__()
        self.name = name
        self.args = args if args is not None else {}

        self.num_training_steps = self.args.get('num_training_steps', DEFAULT_NUM_TRAINING_STEPS)
        beta_start = self.args.get('beta_start', DEFAULT_BETA_START)
        beta_end = self.args.get('beta_end', DEFAULT_BETA_END)
        self.beta_schedule = self.args.get('beta_schedule', DEFAULT_BETA_SCHEDULE)
        self.register_buffer("betas", 
                             make_beta_schedule(schedule=self.beta_schedule, 
                                                num_steps=self.num_training_steps, 
                                                beta_start=beta_start, 
                                                beta_end=beta_end).to(device))
        assert self.betas.shape == (self.num_training_steps,)
        self.rescale_betas_zero_snr = self.args.get('rescale_betas_zero_snr', DEFAULT_RESCALE_BETAS_ZERO_SNR)
        if self.rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        if self.rescale_betas_zero_snr:
            self.alphas_cumprod[-1] = 2 ** -24
        self.init_noise_sigma = 1.0
        self.num_inference_steps = self.args.get('num_inference_steps', DEFAULT_NUM_INFERENCE_STEPS)
        self.timestep_spacing = self.args.get('timestep_spacing', DEFAULT_TIMESTEP_SPACING)
        self.steps_offset = self.args.get('steps_offset', DEFAULT_STEPS_OFFSET)
        self.training_mode = training
        time_steps = torch.linspace(0, 
                                    self.num_training_steps - 1, 
                                    self.num_training_steps).flip(0).long().to(device)
        self.register_buffer("time_steps", time_steps)
        self.sigma_min = self.args.get('sigma_min', DEFAULT_SIGMA_MIN)
        self.sigma_max = self.args.get('sigma_max', DEFAULT_SIGMA_MAX)

    @property
    def device(self) -> torch.device:
        return self.betas.device

    def update_time_steps(self, 
                          num_inference_steps):
        self.num_inference_steps = num_inference_steps
        self.time_steps = update_time_steps(self.num_training_steps, 
                                            self.num_inference_steps, 
                                            self.timestep_spacing, 
                                            self.steps_offset)

    def scale_model_input(self, 
                          sample: torch.Tensor, 
                          time_step: int) -> torch.Tensor:
        return sample

    @abstractmethod
    def sample(self, 
               model_output: torch.Tensor, 
               time_step: int, 
               sample: torch.Tensor, 
               *, 
               eta: float = 0.0, 
               use_clipped_model_output: bool = False, 
               generator: Optional[Union[List[torch.Generator], torch.Generator]] = None, 
               variance_noise: Optional[torch.Tensor] = None, 
               states: dict = None) -> dict:
        ...

    @abstractmethod
    def add_noise(self, 
                  original_samples: torch.Tensor, 
                  noise: torch.Tensor, 
                  time_steps: torch.IntTensor, 
                  *, 
                  returns_velocity: bool = False) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        ...