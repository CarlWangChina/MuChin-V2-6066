import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from einops import rearrange
from Code_for_Experiment.Targeted_Training.dit_training_on_vae.music_dit2.modules.embeddings.rotary_pos_embedding import RotaryPosEmbedding

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_heads: int,
                 hidden_dim: Optional[int] = None,
                 context_dim: Optional[int] = None,
                 qkv_bias: bool = False,
                 rope: Optional[RotaryPosEmbedding] = None,
                 use_rpr: bool = False,
                 max_position: int = 10000,
                 attn_dropout: float = 0.0,
                 proj_dropout: float = 0.0):
        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        hidden_dim = hidden_dim or input_dim
        context_dim = context_dim or input_dim
        assert hidden_dim % num_heads == 0, "dim must be divisible by num_heads."
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.scale = head_dim ** (-0.5)
        self.wq = nn.Linear(input_dim, hidden_dim, bias=qkv_bias)
        self.wk = nn.Linear(context_dim, hidden_dim, bias=qkv_bias)
        self.wv = nn.Linear(context_dim, hidden_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.proj_drop = nn.Dropout(proj_dropout)
        self.rope = rope
        if use_rpr:
            self.rpr = nn.Parameter(torch.randn(max_position, head_dim))
        else:
            self.rpr = None

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.wq.weight)
        nn.init.xavier_uniform_(self.wk.weight)
        nn.init.xavier_uniform_(self.wv.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.rpr is not None:
            nn.init.xavier_uniform_(self.rpr)

    def forward(self,
                x: torch.Tensor,
                *,
                positions: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None,
                context: Optional[torch.Tensor] = None,
                context_positions: Optional[torch.Tensor] = None,
                context_mask: Optional[torch.Tensor] = None,
                causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert x.dim() == 3, "Input tensor must have 3 dimensions, as (batch, seq_len, dim)."
        if context is None:
            context = x
            context_positions = positions
            is_self_attention = True
        else:
            assert context.dim() == 3, "Context tensor must have 3 dimensions, as (batch, seq_len, dim)."
            assert context.size(0) == x.size(0), "Batch size of input and context must be the same."
            is_self_attention = False
        q = self.wq(x)
        k = self.wk(context)
        v = self.wv(context)
        if self.rope is not None:
            assert positions is not None, "Position tensor must be provided when RoPE is used."
            assert positions.size() == (x.size(0), x.size(1)), \
                (f"Position tensor {positions.size()} must have the same shape as input "
                 f"{(x.size(0), x.size(1))}.")
            assert context_positions is not None, "Context position tensor must be provided when RoPE is used."
            assert context_positions.size() == (context.size(0), context.size(1)), \
                (f"Context position tensor {context_positions.size()} must have the same shape as context "
                 f"{(context.size(0), context.size(1))}.")
            q = self.rope(q, positions, mask=padding_mask)
            k = self.rope(k, context_positions, mask=context_mask)
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads).contiguous()
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads).contiguous()
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads).contiguous()
        attn_score = torch.einsum("b h i d, b h j d -> b h i j", q, k).contiguous()
        if self.rpr is not None and is_self_attention and positions is not None:
            rpr_matrix = self.rpr[positions]
            q_embedding = torch.einsum("b h i d, b j d -> b h i j", q, rpr_matrix).contiguous()
            attn_score = attn_score + q_embedding
        if is_self_attention and causal_mask is not None:
            attn_score = attn_score + causal_mask
        attn_score = F.softmax(attn_score * self.scale, dim=-1, dtype=torch.float32).type_as(q)
        if self.training:
            attn_score = self.attn_drop(attn_score)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn_score, v).contiguous()
        out = rearrange(out, "b h s d -> b s (h d)", h=self.num_heads).contiguous()
        out = self.proj(out)
        if self.training:
            out = self.proj_drop(out)
        return out