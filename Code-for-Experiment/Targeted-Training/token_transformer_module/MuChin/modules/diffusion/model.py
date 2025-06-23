from abc import ABC, abstractmethod
import torch.nn as nn
from einops import rearrange
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger
from .unet import UnetModel, UnetModelArgs, ControlNetModel
from .wavenet import WaveNetModelArgs, WaveNetModel
logger = get_logger(__name__)

class DiffusionModel(ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, condition: torch.Tensor = None, context: torch.Tensor = None):
        ...

class UnetDiffusionModel(DiffusionModel):
    def __init__(self, args: UnetModelArgs, device: str = "cpu"):
        super(UnetDiffusionModel, self).__init__()
        self.unet = UnetModel(args).to(device)
        if args.use_controlnet:
            self.controlnet = ControlNetModel(args).to(device)
            logger.info("DiffusionModel: Unet + ControlNet denoiser models initialized.")
        else:
            self.controlnet = None
            logger.info("DiffusionModel: Unet denoiser model initialized without ControlNet.")

    @property
    def device(self):
        return self.unet.device

    def freeze_unet_model(self, freeze: bool = True):
        for param in self.unet.parameters():
            param.requires_grad = not freeze

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, condition: torch.Tensor = None, context: torch.Tensor = None):
        assert x is not None
        assert time_steps is not None
        if condition is not None:
            assert self.controlnet is not None, "ControlNet is not enabled.  Cannot use condition."
            controlnet_output = self.controlnet(x, time_steps=time_steps, control_condition=condition, context=context)
        else:
            controlnet_output = None
        return self.unet(x, time_steps=time_steps, context=context, controlnet_output=controlnet_output)

class WaveNetDiffusionModel(DiffusionModel):
    def __init__(self, args: WaveNetModelArgs, device: str = "cpu"):
        super(WaveNetDiffusionModel, self).__init__()
        self.model = WaveNetModel(args, device=device)
        self.controlnet = None
        logger.info("DiffusionModel: WaveNet denoiser model initialized.")

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, condition: torch.Tensor = None, context: torch.Tensor = None):
        assert x is not None
        assert time_steps is not None
        assert condition is None, "WaveNet denoiser model does not support ControlNet condition."
        if context is not None and context.dim() == 2:
            context = rearrange(context, "b d -> b d 1")
        return self.model(x, time_steps=time_steps, context=context)

    @property
    def device(self):
        return self.model.device