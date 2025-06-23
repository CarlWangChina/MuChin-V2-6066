import torch
from torch import nn
from einops import rearrange
from typing import Optional

from ama_prof_divi.utils import safe_softmax
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.modules.transformers.model_args import TransformerModelArgs
from .acceleration import InferAccelerationCache

class MultiHeadAttention(nn.Module):
    def __init__(self, args: TransformerModelArgs, cross_attention: bool, layer: int, device: str or torch.device = 'cpu'):
        super(MultiHeadAttention, self).__init__()
        self.dim = args.dim
        self.kv_dim = args.kv_dim if args.kv_dim else args.dim
        self.num_heads = args.num_heads
        self.head_dim = args.dim // args.num_heads
        self.max_seq_len = args.max_seq_len
        self.cross_attention = cross_attention
        self.layer = layer
        self.scale = self.head_dim ** (-0.5)
        self.wq = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=False).to(device)
        self.wk = nn.Linear(self.kv_dim, self.num_heads * self.head_dim, bias=False).to(device)
        self.wv = nn.Linear(self.kv_dim, self.num_heads * self.head_dim, bias=False).to(device)
        self.wo = nn.Linear(self.num_heads * self.head_dim, self.dim, bias=False).to(device)

    @property
    def device(self):
        return self.wq.weight.device

    def forward(self, x: torch.Tensor, context_k: torch.Tensor = None, context_v: torch.Tensor = None, mask: torch.Tensor = None, cache: Optional[InferAccelerationCache] = None, start_pos: int = 0) -> torch.Tensor:
        assert x.dim() == 3, f'Input tensor has wrong dimension. Must be (batch, sequence_len, {self.dim} )!'
        assert x.shape[-1] == self.dim, 'Input tensor has wrong dimension. Must be (batch, sequence_len, {self.dim} )!'
        assert context_k.dim() == 3, f'Context tensor has wrong dimension. Must be (batch, context_len, {self.kv_dim} )!'
        assert context_k.shape[-1] == self.kv_dim, f'Context tensor has wrong dimension. Must be (batch, context_len, {self.kv_dim} )!'
        assert x.shape[0] == context_k.shape[0], 'Batch sizes of input and context tensors do not match!'
        assert context_k.shape == context_v.shape, 'The shape of context_k and context_v must be the same.'
        if mask is not None:
            assert mask.dim() == 2, 'Mask tensor has wrong dimension. Must be (seq_len, start_pos + seq_len)!'
            assert mask.shape[0] == x.shape[1], 'Sequence lengths of input and mask tensors do not match!'
            assert mask.shape[1] == context_k.shape[1], 'Context lengths of input and mask tensors do not match!'

        xq = self.wq(x)
        xq = rearrange(xq, 'b s (h d) -> b s h d', h=self.num_heads, d=self.head_dim)

        if self.cross_attention:
            if cache is not None and cache.is_kv_cache_cross_set(self.layer):
                yk, yv = cache.get_kv_cache_cross(self.layer, self.device)
            else:
                yk = self.wk(context_k)
                yv = self.wv(context_v)
                yk = rearrange(yk, 'b s (h d) -> b s h d', h=self.num_heads, d=self.head_dim)
                yv = rearrange(yv, 'b s (h d) -> b s h d', h=self.num_heads, d=self.head_dim)
                if cache is not None:
                    cache.update_kv_cache_cross(self.layer, yk, yv)
        else:
            yk = self.wk(context_k)
            yv = self.wv(context_v)
            yk = rearrange(yk, 'b s (h d) -> b s h d', h=self.num_heads, d=self.head_dim)
            yv = rearrange(yv, 'b s (h d) -> b s h d', h=self.num_heads, d=self.head_dim)
            if cache is not None:
                cache.update_kv_cache_self(self.layer, yk, yv, start_pos)
                yk, yv = cache.get_kv_cache_self(self.layer, 0, start_pos + yk.shape[1], self.device)

        xq = rearrange(xq, 'b s h d -> b h s d')
        yk = rearrange(yk, 'b s h d -> b h s d')
        yv = rearrange(yv, 'b s h d -> b h s d')
        assert xq.device == yk.device, f'Device mismatch! {xq.device} vs. {yk.device} (self.device={self.device} )!'

        scores = torch.einsum("b h q d, b h s d -> b h q s", xq, yk) * self.scale
        if mask is not None:
            scores = scores + mask
        scores = safe_softmax(scores.float(), dim=-1, dtype=torch.float32).type_as(xq)
        output = torch.einsum("b h q s, b h s d -> b h q d", scores, yv)
        output = self.wo(rearrange(output, 'b h s d -> b s (h d)'))
        return output