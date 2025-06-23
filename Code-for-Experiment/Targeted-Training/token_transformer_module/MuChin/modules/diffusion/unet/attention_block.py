import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ama_prof_divi.utils import safe_softmax as safe_softmax_1
from .norm import normalization, zero_module
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.modules.diffusion.unet.conv import conv_nd

class AttentionBlock(nn.Module):
    def __init__(self, dims: int, channels: int, num_heads: int = 1, context_channels: int = -1):
        super(AttentionBlock, self).__init__()
        self.num_channels = channels
        self.num_heads = num_heads
        self.context_channels = context_channels
        self.norm = normalization(channels)
        self.qkv = conv_nd(dims, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        if context_channels > 0:
            self.encoder_kv = conv_nd(dims, context_channels, channels * 2, 1)
        self.proj_out = zero_module(conv_nd(dims, channels, channels, 1))

    def forward(self, x: torch.Tensor, context: torch.Tensor = None, mask: torch.Tensor = None):
        b, c, *spatial = x.shape
        qkv = self.qkv(self.norm(x).view(b, c, -1))
        if context is not None:
            assert self.context_channels > 0
            context = self.encoder_kv(context)
            h = self.attention(qkv, context, mask=mask)
        else:
            h = self.attention(qkv)
        h = self.proj_out(h)
        return x + h.reshape(b, c, *spatial)

class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: torch.Tensor, context: torch.Tensor = None, mask: torch.Tensor = None):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        if context is not None:
            assert context.shape[1] == self.n_heads * ch * 2
            ek, ev = context.reshape(bs * self.n_heads, ch * 2, -1).split(ch, dim=1)
            k = torch.cat([ek, k], dim=-1)
            v = torch.cat([ev, v], dim=-1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct, bcs -> bts", q * scale, k * scale)
        if mask is not None:
            mask = F.pad(mask, (0, length), value=0.0)
            mask = (
                mask.unsqueeze(1)
                .expand(-1, self.n_heads, -1)
                .reshape(bs * self.n_heads, 1, -1)
            )
            weight = weight + mask
        weight = safe_softmax_1(weight, dim=-1)
        a = torch.einsum("bts, bcs -> bct", weight, v)
        return a.reshape(bs, -1, length)