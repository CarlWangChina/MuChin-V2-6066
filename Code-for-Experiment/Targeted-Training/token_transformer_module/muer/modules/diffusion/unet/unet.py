import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from ama_prof_divi.utils.logging import get_logger
from .model_args import UnetModelArgs
from .unet_common import UnetCommon
from .norm import normalization, zero_module
from .conv import conv_nd
from .res_blocks import ResBlock, ResBlockWithTimeEmbedding
from .timestep_block import TimestepEmbedSequential
from .attention_block import AttentionBlock
from .transformer_block import TransformerBlock
from .upsample_block import UpSampleBlock

logger = get_logger(__name__)

@dataclass
class ControlNetOutput:
    down_blocks_outputs: list[torch.Tensor]
    mid_blocks_outputs: list[torch.Tensor]

class UnetModel(nn.Module):
    def __init__(self, args: UnetModelArgs):
        super(UnetModel, self).__init__()
        self.args = args
        time_embedding_dim = self.args.model_channels * 4
        self.common = UnetCommon(args)
        input_block_channels = self.common.input_block_channels
        num_channels = self.common.num_channels_after_init
        ds = self.common.ds_after_init
        res_block_cls = ResBlockWithTimeEmbedding if self.args.use_time_embedding else ResBlock
        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(self.args.channel_mult))[:-1]:
            for i in range(self.args.num_res_blocks + 1):
                ich = input_block_channels.pop()
                orig_num_channels = num_channels
                layers = [
                    res_block_cls(channels=num_channels + ich,
                                  emb_channels=time_embedding_dim,
                                  dropout=self.args.dropout,
                                  out_channels=int(self.args.model_channels * mult),
                                  use_scale_shift_norm=self.args.use_scale_shift_norm,
                                  dims=self.args.dims)
                ]
                num_channels = int(self.args.model_channels * mult)
                if ds in self.args.attention_resolutions:
                    if self.args.use_transformer:
                        layers.append(TransformerBlock(dims=self.args.dims,
                                                       in_channels=num_channels,
                                                       num_heads=self.args.num_heads,
                                                       head_dim=num_channels // self.args.num_heads,
                                                       context_dim=self.args.context_dim,
                                                       depth=self.args.transformer_depth))
                    else:
                        layers.append(AttentionBlock(dims=self.args.dims,
                                                     channels=num_channels,
                                                     num_heads=self.args.num_heads,
                                                     context_channels=self.args.context_dim))
                if level > 0 and i == self.args.num_res_blocks:
                    layers.append(
                        res_block_cls(channels=num_channels,
                                      emb_channels=time_embedding_dim,
                                      dropout=self.args.dropout,
                                      up_sampling=True,
                                      use_scale_shift_norm=self.args.use_scale_shift_norm,
                                      dims=self.args.dims)
                        if self.args.res_block_updown else
                        UpSampleBlock(channels=num_channels,
                                      use_conv=self.args.conv_resample,
                                      dims=self.args.dims)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self.common.block_desc.append(f"Up block [{level}] : {orig_num_channels} + {ich} [{len(input_block_channels)}] -> {num_channels} ({'with' if ds in self.args.attention_resolutions else 'without'} attention)")
                if level > 0 and i == self.args.num_res_blocks:
                    self.common.block_desc.append(f"Up sampling block [{level - 1}] : {num_channels}")
        self.out = nn.Sequential(
            normalization(num_channels),
            nn.SiLU(),
            zero_module(conv_nd(dims=self.args.dims,
                                in_channels=num_channels,
                                out_channels=self.args.out_channels,
                                kernel_size=3,
                                padding=1))
        )
        self.common.block_desc.append(f"Final output block: {num_channels} -> {self.args.out_channels}")
        logger.info("Unet model generated.")

    @property
    def input_blocks(self):
        return self.common.input_blocks

    @property
    def middle_blocks(self):
        return self.common.middle_blocks

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def block_desc(self):
        return self.common.block_desc

    def forward(self, x: torch.Tensor, *, time_steps: torch.Tensor, context: Optional[torch.Tensor] = None, controlnet_output: Optional[ControlNetOutput] = None):
        x, time_emb, context = self.common.preprocess_input(x, time_steps, context)
        x_stack = []
        for module in self.input_blocks:
            x = module(x, time_emb, context)
            x_stack.append(x.cpu())
        x = self.middle_blocks(x, time_emb, context)
        if controlnet_output is not None:
            x += controlnet_output.mid_blocks_outputs.pop().to(x.device)
        for module in self.output_blocks:
            if controlnet_output is not None:
                x = torch.cat([x, x_stack.pop().to(x.device) + controlnet_output.down_blocks_outputs.pop().to(x.device)], dim=1)
            else:
                x = torch.cat([x, x_stack.pop().to(x.device)], dim=1)
            x = module(x, time_emb, context)
        return self.out(x)