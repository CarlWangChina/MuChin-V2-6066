import torch
import torch.nn as nn
from einops import rearrange
from ama_prof_divi.utils.logging import get_logger
from .model_args import UnetModelArgs
import UnetModelArgs
from .res_blocks import ResBlock, ResBlockWithTimeEmbedding
import ResBlock, ResBlockWithTimeEmbedding
from .conv import conv_nd
import conv_nd
from .embeddings import timestep_embedding
import timestep_embedding
from .timestep_block import TimestepEmbedSequential
import TimestepEmbedSequential
from .attention_block import AttentionBlock
import AttentionBlock
from .transformer_block import TransformerBlock
import TransformerBlock
import DownSampleBlock
logger = get_logger(__name__)

class UnetCommon(nn.Module):
    def __init__(self, args: UnetModelArgs):
        super(UnetCommon, self).__init__()
        self.args = args
        if self.args.use_transformer:
            assert self.args.transformer_depth > 0, "transformer_depth must be greater than 0 if use_transformer is True."
        time_embedding_dim = self.args.model_channels * 4
        if self.args.use_time_embedding:
            self.time_embedding = nn.Sequential(
                nn.Linear(self.args.model_channels, time_embedding_dim),
                nn.SiLU(),
                nn.Linear(time_embedding_dim, time_embedding_dim),
            )
        else:
            self.time_embedding = None
        self.block_desc = []

        num_channels = int(self.args.model_channels * self.args.channel_mult[0])

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims=self.args.dims,
                                             in_channels=num_channels,
                                             out_channels=num_channels,
                                             kernel_size=3,
                                             padding=1))]
        )
        self.block_desc.append(f"Input block: {num_channels} -> {num_channels} [0]")
        res_block_cls = ResBlockWithTimeEmbedding if self.args.use_time_embedding else ResBlock
        self.input_block_channels = [num_channels]

        ds = 1
        for level, mult in enumerate(self.args.channel_mult):
            for _ in range(self.args.num_res_blocks):
                orig_num_channels = num_channels
                layers = [
                    res_block_cls(channels=num_channels,
                                  emb_channels=time_embedding_dim,
                                  dropout=self.args.dropout,
                                  out_channels=int(self.args.model_channels * mult),
                                  use_scale_shift_norm=self.args.use_scale_shift_norm,
                                  dims=self.args.dims)]

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
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.input_block_channels.append(num_channels)
                self.block_desc.append(f"Down block [{level}] : {orig_num_channels} -> {num_channels} [{len(self.input_block_channels) - 1}] ({'with' if ds in self.args.attention_resolutions else 'without'} attention) ")
            if level != len(self.args.channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        res_block_cls(channels=num_channels,
                                      emb_channels=time_embedding_dim,
                                      dropout=self.args.dropout,
                                      down_sampling=True,
                                      use_scale_shift_norm=self.args.use_scale_shift_norm,
                                      dims=self.args.dims)
                        if self.args.res_block_updown else
                        DownSampleBlock(channels=num_channels,
                                        use_conv=self.args.conv_resample,
                                        dims=self.args.dims)
                    )
                )
                self.input_block_channels.append(num_channels)
                self.block_desc.append(f"Down sampling block [{level}] : {num_channels} [{len(self.input_block_channels) - 1}]")
                ds *= 2
        self.middle_blocks = TimestepEmbedSequential(
            res_block_cls(channels=num_channels,
                          emb_channels=time_embedding_dim,
                          dropout=self.args.dropout,
                          use_scale_shift_norm=self.args.use_scale_shift_norm,
                          dims=self.args.dims),
            TransformerBlock(dims=self.args.dims,
                             in_channels=num_channels,
                             num_heads=self.args.num_heads,
                             head_dim=num_channels // self.args.num_heads,
                             context_dim=self.args.context_dim,
                             depth=self.args.transformer_depth) if self.args.use_transformer else
            AttentionBlock(dims=self.args.dims,
                           channels=num_channels,
                           num_heads=self.args.num_heads,
                           context_channels=self.args.context_dim),
            res_block_cls(channels=num_channels,
                          emb_channels=time_embedding_dim,
                          dropout=self.args.dropout,
                          use_scale_shift_norm=self.args.use_scale_shift_norm,
                          dims=self.args.dims)
        )
        self.num_channels_after_init = num_channels
        self.ds_after_init = ds
        self.block_desc.append(f"Middle block: {num_channels} -> {num_channels}")

    def preprocess_input(self, x: torch.Tensor, time_steps: torch.Tensor, context: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        assert x is not None, "Input tensor x cannot be None."
        assert x.dim() == 3
        seq_len = x.shape[-1]

        assert seq_len % (2 ** len(self.args.channel_mult)) == 0, f"The seq length ({seq_len}) must be divisible by 2 ** len(channel_mult), i.e. {2 ** len(self.args.channel_mult)}."
        if self.args.use_time_embedding is not None:
            assert time_steps is not None, "Time steps cannot be None."
            assert time_steps.dim() == 1, "Time steps must be a 1D tensor."
            assert time_steps.shape[0] == x.shape[0], "Time steps must have the same batch size as the input tensor. TimeSteps: {}, Input: {}".format(
                time_steps.shape[0], x.shape[0])
            time_emb = timestep_embedding(time_steps, self.args.model_channels, repeat_only=False)
            time_emb = time_emb.to(next(self.time_embedding.parameters()).dtype)
            time_emb = self.time_embedding(time_emb)
        else:
            time_emb = None
        if context is not None:
            assert context.dim() == 2 or context.dim() == 3, "Context must be a 3D tensor."
            assert context.shape[0] == x.shape[0], "Context must have the same batch size as the input tensor."
            assert self.args.context_dim == context.shape[1], "Context must have the same dimension as the context_dim argument."
            if context.dim() == 2:
                context = rearrange(context, 'b c -> b c 1')
            if self.args.use_transformer:
                context = rearrange(context, 'b c ... -> b ... c')
        return x, time_emb, context