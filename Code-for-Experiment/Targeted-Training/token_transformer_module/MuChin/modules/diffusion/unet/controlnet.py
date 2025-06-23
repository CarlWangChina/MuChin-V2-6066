import torch
import torch.nn as nn
from ama_prof_divi.utils.logging import get_logger
from .model_args import UnetModelArgs
import UnetModel
from .norm import zero_module
from .conv import conv_nd
import TimestepEmbedSequential
from .unet_common import UnetCommon
from .unet import UnetModel, ControlNetOutput
logger = get_logger(__name__)

def _make_zero_conv(dims: int, channels: int):
    return TimestepEmbedSequential(zero_module(conv_nd(dims=dims,
                                                       in_channels=channels,
                                                       out_channels=channels,
                                                       kernel_size=1,
                                                       padding=0)))

class ControlNetModel(nn.Module):
    def __init__(self, args: UnetModelArgs):
        super(ControlNetModel, self).__init__()
        self.args = args
        self.common = UnetCommon(args)
        num_channels = int(self.args.model_channels * self.args.channel_mult[0])
        self.conditioning_input_block = TimestepEmbedSequential(
            conv_nd(dims=self.args.dims,
                    in_channels=args.in_channels,
                    out_channels=num_channels,
                    kernel_size=3,
                    padding=1),
            nn.SiLU(),
            zero_module(conv_nd(dims=self.args.dims,
                                in_channels=num_channels,
                                out_channels=num_channels,
                                kernel_size=3,
                                padding=1))
        )
        self.control_blocks = nn.ModuleList([
            _make_zero_conv(dims=self.args.dims, channels=num_channels)
        ])
        for level, mult in enumerate(self.args.channel_mult):
            num_channels = int(self.args.model_channels * mult)
            for _ in range(self.args.num_res_blocks):
                self.control_blocks.append(
                    _make_zero_conv(dims=self.args.dims, channels=num_channels)
                )
            if level != len(self.args.channel_mult) - 1:
                self.control_blocks.append(
                    _make_zero_conv(dims=self.args.dims, channels=num_channels)
                )
        self.control_block_middle = _make_zero_conv(dims=self.args.dims, channels=num_channels)
        assert len(self.control_blocks) == len(self.input_blocks) == len(self.common.input_block_channels)
        logger.info("ControlNetModel: Control network initialized.")

    def copy_states_from_unet(self, unet: UnetModel):
        self.common.load_state_dict(unet.common.state_dict())
        logger.info("ControlNetModel: copied trainable states from unet model.")

    @property
    def input_blocks(self):
        return self.common.input_blocks

    @property
    def middle_blocks(self):
        return self.common.middle_blocks

    def forward(self, x: torch.Tensor, *, time_steps: torch.Tensor, control_condition: torch.Tensor, context: torch.Tensor = None) -> ControlNetOutput:
        x, time_emb, context = self.common.preprocess_input(x, time_steps, context)
        condition = self.conditioning_input_block(control_condition, time_emb, context)
        down_blocks_outputs = []
        mid_blocks_outputs = []
        for input_block, control_block in zip(self.input_blocks, self.control_blocks):
            x = input_block(x, time_emb, context)
            if condition is not None:
                x = x + condition
            condition = None
            down_blocks_outputs.append(control_block(x, time_emb, context).cpu())
        x = self.middle_blocks(x, time_emb, context)
        mid_blocks_outputs.append(self.control_block_middle(x, time_emb, context).cpu())
        return ControlNetOutput(down_blocks_outputs, mid_blocks_outputs)