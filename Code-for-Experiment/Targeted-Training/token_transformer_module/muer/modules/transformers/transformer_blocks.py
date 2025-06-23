import torch
import torch.nn as nn
from typing import Optional
from .model_args import TransformerModelArgs
from .acceleration import InferAccelerationCache
from .attention import MultiHeadAttention
from .feed_forward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, args: TransformerModelArgs, layer: int, device: str or torch.device = "cpu"):
        super(TransformerBlock, self).__init__()
        self.device = device
        self.self_attn = MultiHeadAttention(args, cross_attention=False, layer=layer, device=device)
        self.cross_attn = MultiHeadAttention(args, cross_attention=True, layer=layer, device=device)
        self.ff1 = FeedForward(dim=args.dim, hidden_dim=args.hidden_dim, dropout=args.dropout, device=device)
        self.ff2 = FeedForward(dim=args.dim, hidden_dim=args.hidden_dim, dropout=args.dropout, device=device)
        self.ln1 = nn.LayerNorm(args.dim).to(device)
        self.ln2 = nn.LayerNorm(args.dim).to(device)
        self.ln3 = nn.LayerNorm(args.dim).to(device)
        self.ln4 = nn.LayerNorm(args.dim).to(device)

    def forward(self, x: torch.Tensor, *, self_context_k: torch.Tensor, self_context_v: torch.Tensor, cross_context_k: torch.Tensor = None, cross_context_v: torch.Tensor = None, cross_attention: bool = False, mask: Optional[torch.Tensor] = None, cache: Optional[InferAccelerationCache] = None, start_pos: int = 0) -> torch.Tensor:
        x = x + self.self_attn(self.ln1(x), context_k=self_context_k, context_v=self_context_v, start_pos=start_pos, cache=cache, mask=mask)
        x += self.ff1(self.ln2(x))
        if cross_attention:
            x += self.cross_attn(self.ln3(x), context_k=cross_context_k, context_v=cross_context_v, cache=cache)
            x += self.ff2(self.ln4(x))
        return x