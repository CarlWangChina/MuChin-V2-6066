import torch
import torch.nn as nn
from einops import rearrange, repeat
from ama_prof_divi.utils import safe_softmax as safe_softmax_1
from ama_prof_divi.utils.logging import get_logger
from .norm import normalization, zero_module
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.modules.diffusion.unet.conv import conv_nd
_1_logger = get_logger(__name__)

class SpatialSelfAttention(nn.Module):
    def __init__(self, dims: int, in_channels: int):
        super(SpatialSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.norm = normalization(in_channels)
        self.cq = conv_nd(dims, in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.ck = conv_nd(dims, in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.cv = conv_nd(dims, in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = conv_nd(dims, in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        assert x.dim() == 3, "The input tensor must have at 3 dimensions (batch, frame, channels)."
        num_channels = x.shape[2]
        assert num_channels == self.in_channels
        x = rearrange(x, 'b ... c -> b c ...')
        h = self.norm(x)
        q = self.cq(h)
        k = self.ck(h)
        v = self.cv(h)
        h = torch.einsum('b c t, b c s -> b t s', q, k) * (num_channels ** (-0.5))
        h = safe_softmax_1(h, dim=-1)
        h = torch.einsum('b t s, b c s -> b c t', h, v)
        h = self.proj_out(h)
        return rearrange(x + h, 'b c ... -> b ... c')

class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, context_dim: int, dropout: float = 0.0):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.context_dim = context_dim if context_dim > 0 else dim
        inner_dim = num_heads * head_dim
        self.wq = nn.Linear(self.dim, inner_dim, bias=False)
        self.wk = nn.Linear(self.context_dim, inner_dim, bias=False)
        self.wv = nn.Linear(self.context_dim, inner_dim, bias=False)
        self.wo = nn.Linear(inner_dim, self.dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = head_dim ** (-0.5)

    def forward(self, x: torch.Tensor, context: torch.Tensor, mask: torch.Tensor = None):
        if context is None:
            context = x
        assert x.dim() == 3, "The input tensor must have at 3 dimensions (batch, frame, dim)."
        assert context.dim() == 3, "The context tensor must have at 3 dimensions (batch, frame, context_dim)."
        assert x.shape[0] == context.shape[0], "The batch size of the input tensor and the context tensor must be the same."
        assert x.shape[2] == self.dim, f"x shape: {x.shape}, dim: {self.dim}"
        assert context.shape[2] == self.context_dim, f"context shape: {context.shape}, context_dim: {self.context_dim}"
        q = self.wq(x)
        k = self.wk(context)
        v = self.wv(context)
        q = rearrange(q, 'b f (h d) -> b h f d', h=self.num_heads)
        k = rearrange(k, 'b f (h d) -> b h f d', h=self.num_heads)
        v = rearrange(v, 'b f (h d) -> b h f d', h=self.num_heads)
        scores = torch.einsum("b h q d, b h s d -> b h q s", q, k) * self.scale
        del q, k
        if mask is not None:
            mask = repeat(mask, 'b s -> b h q s', h=self.num_heads, q=x.shape[1])
            scores += mask
        scores = safe_softmax_1(scores, dim=-1)
        scores = torch.einsum('b h q s, b h s d -> b h q d', scores, v)
        scores = rearrange(scores, 'b h q d -> b q (h d)')
        return self.dropout(self.wo(scores))

class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)

class TransformerBlockInnerLayer(nn.Module):
    def __init__(self, dims: int, hidden_dim: int, in_channels: int, num_heads: int, head_dim: int, context_dim: int, dropout: float = 0.0, disable_self_attention: bool = False):
        super(TransformerBlockInnerLayer, self).__init__()
        self.disable_self_attention = disable_self_attention
        self.in_channels = in_channels
        self.self_attention = SpatialSelfAttention(dims=dims, in_channels=in_channels) if not self.disable_self_attention else None
        self.ff = FeedForward(hidden_dim, dropout=dropout)
        self.cross_attention = CrossAttention(dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, context_dim=context_dim, dropout=dropout)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim, eps=1e-4), nn.LayerNorm(hidden_dim, eps=1e-4), nn.LayerNorm(hidden_dim, eps=1e-4)])

    def forward(self, x: torch.Tensor, context: torch.Tensor = None, mask: torch.Tensor = None):
        assert x.dim() == 3, "The input tensor must have at 3 dimensions (batch, frame, channels)."
        assert x.shape[2] == self.in_channels, f"x shape: {x.shape}, in_channels: {self.in_channels}"
        if not self.disable_self_attention:
            x += self.self_attention(self.norms[0](x))
        else:
            x = x.clone()
        x += self.cross_attention(self.norms[1](x), context, mask)
        x += self.ff(self.norms[2](x))
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dims: int, in_channels: int, num_heads: int, head_dim: int, depth: int, context_dim: int, dropout: float = 0.0, disable_self_attention: bool = False, use_linear_attention: bool = False):
        super(TransformerBlock, self).__init__()
        self.use_linear_attention = use_linear_attention
        self.in_channels = in_channels
        self.context_dim = context_dim
        hidden_dim = head_dim * num_heads
        self.norm = normalization(in_channels)
        self.proj_in = nn.Linear(in_channels, hidden_dim) if self.use_linear_attention else conv_nd(dims, in_channels, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.blocks = nn.ModuleList([
            TransformerBlockInnerLayer(dims=dims, in_channels=in_channels, hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, context_dim=context_dim, dropout=dropout, disable_self_attention=disable_self_attention)
            for _ in range(depth)
        ])
        self.proj_out = zero_module(nn.Linear(hidden_dim, in_channels) if self.use_linear_attention else conv_nd(dims, hidden_dim, in_channels, kernel_size=1, stride=1, padding=0))

    def forward(self, x: torch.Tensor, context: torch.Tensor = None, mask: torch.Tensor = None):
        assert x.dim() == 3, "The input tensor must have at 3 dimensions."
        if context is not None:
            assert context.dim() == 3, "The context tensor must have at 3 dimensions."
            assert x.shape[0] == context.shape[0], "The batch size of the input tensor and the context tensor must be the same."
            assert self.context_dim == context.shape[2], "The frame size of the context tensor must be context_dim."
        if mask is not None:
            assert context is not None, "The context tensor must be provided if the mask tensor is provided."
            assert mask.dim() == 2, "The mask tensor must have at 2 dimensions."
            assert x.shape[0] == mask.shape[0], "The batch size of the input tensor and the mask tensor must be the same."
            assert self.context_dim == mask.shape[1], "The context channel size of the input tensor and the mask tensor must be the same."
        num_channels = x.shape[1]
        assert num_channels == self.in_channels, f"x shape: {x.shape}, in_channels: {self.in_channels}"
        h = self.norm(x)
        if self.use_linear_attention:
            h = rearrange(h, 'b c ... -> b ... c')
            h = self.proj_in(h)
        else:
            h = self.proj_in(h)
            h = rearrange(h, 'b c ... -> b ... c')
        for block in self.blocks:
            h = block(h, context, mask)
        if self.use_linear_attention:
            h = self.proj_out(h)
            h = rearrange(h, 'b ... c -> b c ...')
        else:
            h = rearrange(h, 'b ... c -> b c ...')
            h = self.proj_out(h)
        return x + h