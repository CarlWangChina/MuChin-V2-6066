import torch
from torch import nn
from typing import Optional
from einops import rearrange
from ama_prof_divi.utils.logging import get_logger
from .model_args import TransformerModelArgs
import TransformerModelArgs
import TransformerBlock
from .acceleration import InferAccelerationCache
import InferAccelerationCache
from ..embeddings import RotaryPosEmbedding, SinusoidalPositionalEmbedding

logger = get_logger(__name__)

class TransformerBase(nn.Module):
    def __init__(self, args: TransformerModelArgs, device: str or torch.device = "cpu"):
        super(TransformerBase, self).__init__()
        self.dim = args.dim
        self.vocab_size = args.vocab_size
        self.max_seq_len = args.max_seq_len
        self.num_quantization_groups = args.num_quantization_groups
        assert self.dim > 0, f"Model dimension must be positive, got {self.dim}."
        assert self.vocab_size > 0, f"Vocabulary size must be positive, got {self.vocab_size}."
        assert self.max_seq_len > 0, f"Maximum sequence length must be positive, got {self.max_seq_len}."
        assert self.dim % args.num_heads == 0, \
            f"Model dimension {self.dim} must be divisible by number of heads {args.num_heads}."
        self.embeddings = nn.ModuleList([
            nn.Embedding(self.vocab_size, self.dim, device=device) for _ in range(self.num_quantization_groups)
        ])
        max_position_embeddings = args.max_position_embeddings
        if max_position_embeddings is None:
            max_position_embeddings = self.max_seq_len
        if max_position_embeddings < self.max_seq_len:
            max_position_embeddings = self.max_seq_len
        assert args.pos_embedding in ["rotary", "sinusoidal", "none"], \
            (f"Positional embedding type {args.pos_embedding} is not supported.  "
             f"Must be either 'rotary' or 'sinusoidal'.")
        if args.pos_embedding == "rotary":
            self.pos_enc = RotaryPosEmbedding(self.dim,
                                              max_seq_len=max_position_embeddings,
                                              device=device)
        elif args.pos_embedding == "sinusoidal":
            self.pos_enc = SinusoidalPositionalEmbedding(self.dim,
                                                         init_size=max_position_embeddings,
                                                         device=device)
        else:
            logger.warning("Positional embedding is disabled by configuration.")
            self.pos_enc = None
        self.ln = nn.LayerNorm(self.dim).to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def _validate_input(self, x: torch.Tensor):
        assert x is not None
        assert x.dtype == torch.long or x.dtype == torch.int, \
            "Input tensor must be of type torch.long or torch.int."
        assert x.dim() == 2 or x.dim() == 3, ("Input tensor must have shape (batch_size, seq_len), or "
                                              "(batch_size, seq_len, num_quantization_groups).")
        assert x.shape[1] <= self.max_seq_len, \
            f"Input sequence length {x.shape[1]} exceeds maximum sequence length {self.max_seq_len}."
        if x.dim() == 3:
            assert x.shape[2] == self.num_quantization_groups, \
                f"Number of quantization groups {x.shape[2]} does not match model number of quantization groups {self.num_quantization_groups}."
        else:
            assert self.num_quantization_groups == 1, \
                f"Number of quantization groups {self.num_quantization_groups} must be 1 when x is 2-dimensional."

    def _validate_context(self, context: Optional[torch.Tensor]):
        assert context is not None
        assert context.dim() == 3, "Context tensor must have shape (batch_size, seq_len, dim)."
        assert context.shape[1] <= self.max_seq_len, \
            f"Context sequence length {context.shape[1]} exceeds maximum sequence length {self.max_seq_len}."
        assert context.shape[2] == self.dim, \
            f"Context dimension {context.shape[2]} does not match model dimension {self.dim}."

    def embedding(self, x: torch.Tensor):
        if x.dim() == 2:
            x = torch.unsqueeze(x, -1)
        assert torch.min(x) >= 0, "Input tensor must have non-negative values."
        assert torch.max(x) < self.vocab_size, f"Input tensor must have values less than {self.vocab_size}."
        x = torch.unsqueeze(x, -1)
        x = torch.cat([self.embeddings[i](x[:, :, i, :].long().to(self.device))
                       for i in range(self.num_quantization_groups)],
                      dim=-2)
        x = torch.sum(x, dim=-2)
        return x

    def _apply_embedding(self, x: torch.Tensor, *, emb: Optional[torch.Tensor] = None,
                         context: Optional[torch.Tensor] = None, use_pos_embedding: bool = True,
                         start_pos: int = 0, pos_bias: int or [int] or torch.Tensor = 0,
                         pos_bias_k: int or [int] or torch.Tensor = 0) -> dict:
        self._validate_input(x)
        x = self.embedding(x)
        self_context_v = x
        if emb is not None:
            emb = emb.to(x.device)
            if type(pos_bias) is list or torch.is_tensor(pos_bias):
                emb_list = [emb[b, pos_bias[b]:pos_bias[b] + x.shape[1]].unsqueeze()
                            for b in range(emb.shape[0])]
                emb = torch.cat(emb_list, dim=0)
            else:
                emb = emb[:, pos_bias:pos_bias + x.shape[1]]
            x = x + emb
        if use_pos_embedding and self.pos_enc is not None:
            x = self.pos_enc.embed_on_transformer(x,
                                                  start_pos=start_pos,
                                                  pos_bias=pos_bias)
        self_context_k = x
        cross_attention = False
        cross_context_k = None
        cross_context_v = None
        if context is not None:
            self._validate_context(context)
            cross_attention = True
            cross_context_k = context.to(self.device)
            cross_context_v = context.to(self.device)
            cross_context_k = cross_context_k + self.pos_enc(cross_context_k, pos_bias=pos_bias_k)
        return {
            'x': x,
            'self_context_k': self_context_k,
            'self_context_v': self_context_v,
            'cross_context_k': cross_context_k,
            'cross_context_v': cross_context_v,
            'cross_attention': cross_attention
        }


class TransformerEncoder(TransformerBase):
    def __init__(self, args: TransformerModelArgs, output_dim: Optional[int] = None,
                 device: str or torch.device = "cpu"):
        super(TransformerEncoder, self).__init__(args, device=device)
        if output_dim is None:
            output_dim = args.dim
        self.out_dim = output_dim
        self.blocks = nn.ModuleList([
            TransformerBlock(args, layer=layer, device=device) for layer in range(args.num_layers)
        ])
        self.output_proj = nn.Linear(args.dim, output_dim).to(device)

    def forward(self, x: torch.Tensor, *, emb: Optional[torch.Tensor] = None,
                context: Optional[torch.Tensor] = None, pos_bias: int or [int] or torch.Tensor = 0,
                pos_bias_k: int or [int] or torch.Tensor = 0) -> torch.Tensor:
        embedded = self._apply_embedding(x,
                                         emb=emb,
                                         context=context,
                                         pos_bias=pos_bias,
                                         pos_bias_k=pos_bias_k)
        for block in self.blocks:
            x = block(**embedded)
        x = self.output_proj(self.ln(x))
        return x


class TransformerDecoder(TransformerBase):
    def __init__(self, args: TransformerModelArgs, output_dim: Optional[int] = None,
                 device: str or torch.device = "cpu"):
        super(TransformerDecoder, self).__init__(args, device=device)
        if output_dim is None:
            output_dim = args.vocab_size
        self.out_dim = output_dim
        self.blocks = nn.ModuleList([
            TransformerBlock(args, layer=layer, device=device) for layer in range(args.num_layers)
        ])
        self.output_proj = nn.Linear(args.dim, output_dim * self.num_quantization_groups).to(device)

    def forward(self, x: torch.Tensor, *, emb: Optional[torch.Tensor] = None,
                context: Optional[torch.Tensor] = None, cache: Optional[InferAccelerationCache] = None,
                start_pos: int = 0, pos_bias: int or [int] or torch.Tensor = 0,
                pos_bias_k: int or [int] or torch.Tensor = 0) -> torch.Tensor:
        embedded = self._apply_embedding(x,
                                         emb=emb,
                                         context=context,
                                         start_pos=start_pos,
                                         pos_bias=pos_bias,
                                         pos_bias_k=pos_bias_k)
        seqlen = x.shape[1]
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float('-inf'), device=self.device)
            mask = torch.triu(mask, diagonal=1)
            if cache is not None:
                mask = torch.hstack([torch.zeros((seqlen, start_pos), device=self.device),
                                     mask])
        for block in self.blocks:
            x = block(**embedded,
                      mask=mask,
                      cache=cache,
                      start_pos=start_pos)
        x = self.output_proj(self.ln(x))
        if self.num_quantization_groups > 1:
            x = rearrange(x, 'b s (q d) -> b s q d', q=self.num_quantization_groups)
        return x