import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from einops import rearrange
from ama_prof_divi.utils import safe_softmax
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.logging import get_logger
logger = get_logger(__name__)
DEFAULT_MAX_TARGET_POSITIONS = 2000

class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, k_dim: Optional[int] = None, v_dim: Optional[int] = None, dropout: float = 0.1, bias: bool = True, bias_kv: Optional[bool] = None, self_attention: bool = False, encoder_decoder_attention: bool = False, device: str = "cpu"):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.k_dim = k_dim if k_dim is not None else dim
        self.v_dim = v_dim if v_dim is not None else dim
        if self_attention:
            assert (self.k_dim == self.v_dim and self.k_dim == self.dim), 'Self-attention requires query, key and value to be of the same size'
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = self.dim // self.num_heads
        assert self.head_dim * self.num_heads == self.dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.encoder_decoder_attention = encoder_decoder_attention
        self.device = device
        self.in_proj_q = nn.Linear(self.dim, self.dim, bias=bias, device=device)
        self.in_proj_k = nn.Linear(self.k_dim, self.dim, bias=bias if bias_kv is None else bias_kv, device=device)
        self.in_proj_v = nn.Linear(self.v_dim, self.dim, bias=bias if bias_kv is None else bias_kv, device=device)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=bias).to(device)

    def forward(self, query: torch.Tensor, key: Optional[torch.Tensor] = None, value: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert query.dim() == 3, f'Input tensor has wrong dimension. Must be (batch, sequence_len, {self.dim})!'
        assert query.shape[-1] == self.dim, 'Input tensor has wrong dimension. Must be (batch, sequence_len, ' \
                                            f'{self.dim})'
        if key is None:
            key = query
        else:
            assert key.dim() == 3, f'Input tensor has wrong dimension. Must be (batch, sequence_len, {self.k_dim})!'
            assert key.shape[-1] == self.k_dim, 'Input tensor has wrong dimension. Must be (batch, sequence_len, ' \
                                                f'{self.k_dim})!'
        if value is None:
            value = query
        else:
            assert value.dim() == 3, f'Input tensor has wrong dimension. Must be (batch, sequence_len, {self.v_dim})!'
            assert value.shape[-1] == self.v_dim, 'Input tensor has wrong dimension. Must be (batch, sequence_len, ' \
                                                  f'{self.v_dim})!'
        assert key.shape == value.shape, 'The shape of key and value must be the same.'
        assert key.shape[1] == query.shape[1], 'The batch size of query and key must be the same.'
        xq = self.in_proj_q(query)
        xq = rearrange(xq, "t b (h d) -> t b h d", h=self.num_heads, d=self.head_dim)
        xk = self.in_proj_k(key)
        xk = rearrange(xk, "t b (h d) -> t b h d", h=self.num_heads, d=self.head_dim)
        if key_padding_mask is not None:
            assert key_padding_mask.shape[0:2] == (key.shape[0], key.shape[1]), \
                   (f"The shape of key_padding_mask ({key_padding_mask.shape}) must be (batch_size, seq_len) "
                    f"= ({key.shape[0]}, {key.shape[1]})!")
            key_padding_mask = key_padding_mask.unsqueeze(-1)
            key_padding_mask = key_padding_mask.expand(xk.shape)
            xk *= key_padding_mask
        xv = self.in_proj_v(value)
        xv = rearrange(xv, "t b (h d) -> t b h d", h=self.num_heads, d=self.head_dim)
        scores = torch.einsum("s b h d, t b h d -> h b s t", xq, xk) * self.scaling
        if attention_mask is not None:
            scores *= attention_mask
        scores = safe_softmax(scores, dim=-1)
        output = torch.einsum("h b s t, t b h d -> s b h d", scores, xv)
        output = rearrange(output, "s b h d -> s b (h d)")
        output = self.out_proj(output)
        return output

class TransformFFNLayer(nn.Module):
    def __init__(self, dim: int, filter_size: int, kernel_size: int, dropout: float = 0.1, bias: bool = True, padding_type: str = "same", act_type: str = "gelu", device: str = "cpu"):
        super(TransformFFNLayer, self).__init__()
        self.device = device
        self.dim = dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        assert act_type in ["relu", "gelu"], (f"Unknown activation type: {act_type}.  "
                                              f"Should be either \"relu\" or \"gelu\".")
        self.activation = F.relu if act_type == "relu" else F.gelu
        assert padding_type in ["same", "left"], (f"Unknown padding type: {padding_type}.  "
                                                  f"Should be either \"same\" or \"left\".")
        if padding_type == "same":
            self.ffn1 = nn.Conv1d(in_channels=self.dim, out_channels=filter_size, kernel_size=self.kernel_size, padding=self.kernel_size // 2, device=device)
        else:
            self.ffn1 = nn.Sequential(
                nn.ConstantPad1d(padding=(self.kernel_size - 1, 0), value=0.0),
                nn.Conv1d(in_channels=self.dim, out_channels=filter_size, kernel_size=self.kernel_size, padding=0, device=device)
            )
        self.ffn2 = nn.Linear(filter_size, self.dim, bias=bias, device=device)

    def forward(self, x: torch.Tensor, incremental_state: bool) -> torch.Tensor:
        assert x.dim() == 3, f'Input tensor has wrong dimension. Must be (sequence_len, batch_size, {self.dim})!'
        x = rearrange(x, "t b d -> b d t")
        x = self.ffn1(x)
        x = rearrange(x, "b d t -> t b d")
        x = x * (self.kernel_size ** -0.5)
        if incremental_state:
            x = x[-1:]
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.ffn2(x)
        return x

class SelfAttentionEncoderLayer(nn.Module):
    def __init__(self, dim: int, kernel_size: int, num_heads: int, dropout: float = 0.1, bias: bool = False, norm_type: str = "layer_norm", padding_type: str = "same", act_type: str = "gelu", device: str = "cpu"):
        super(SelfAttentionEncoderLayer, self).__init__()
        assert norm_type in ["layer_norm", "batch_norm"], (f"Unknown normalization: {norm_type}.  "
                                                           f"Should be either \"layer_norm\" or \"batch_norm\".")
        self.dim = dim
        self.num_heads = num_heads
        self.norm_type = norm_type
        self.dropout = dropout
        if self.num_heads > 0:
            if norm_type == "layer_norm":
                self.norm = nn.LayerNorm(self.dim, device=device)
            else:
                self.norm = nn.BatchNorm1d(self.dim, device=device)
            self.self_attn = MultiHeadAttention(dim=self.dim, num_heads=self.num_heads, dropout=dropout, bias=bias, device=device)
        if norm_type == "layer_norm":
            self.ffn_norm = nn.LayerNorm(self.dim, device=device)
        else:
            self.ffn_norm = nn.BatchNorm1d(self.dim, device=device)
        self.ffn = TransformFFNLayer(dim=self.dim, filter_size=4 * self.dim, kernel_size=kernel_size, dropout=dropout, bias=bias, padding_type=padding_type, act_type=act_type, device=device)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, incremental_state: bool = False, norm_training: Optional[bool] = None) -> torch.Tensor:
        if norm_training is not None:
            self.norm.train(mode=norm_training)
            self.ffn_norm.train(mode=norm_training)
        if padding_mask is not None:
            x_padding_mask = padding_mask.expand(-1, -1, x.shape[-1])
        else:
            x_padding_mask = 1.0
        if self.num_heads > 0:
            residual = x
            x = self.norm(x)
            x = self.self_attn(query=x, key_padding_mask=padding_mask, attention_mask=attention_mask)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = x * x_padding_mask
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x, incremental_state=incremental_state)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = x * x_padding_mask
        return x

class FFTBlocks(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, kernel_size: int, num_heads: int, dropout: float = 0.1, use_pos_embedding: bool = True, use_pos_embedding_alpha: bool = True, use_last_norm: bool = True, norm_type: str = "layer_norm", padding_type: str = "same", act_type: str = "gelu", device: str = "cpu"):
        super(FFTBlocks, self).__init__()
        assert norm_type in ["layer_norm", "batch_norm"], (f"Unknown normalization: {norm_type}.  "
                                                           f"Should be either \"layer_norm\" or \"batch_norm\".")
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.use_pos_embedding = use_pos_embedding
        self.use_last_norm = use_last_norm
        self.norm_type = norm_type
        self.device = device
        if self.use_pos_embedding:
            self.max_positions = DEFAULT_MAX_TARGET_POSITIONS
            self.padding_idx = 0
            self.pos_embed_alpha = nn.Parameter(torch.Tensor([1.0]).to(self.device)) if use_pos_embedding_alpha else 1.0
            import SinusoidalPositionalEmbedding
            self.position_embedding = SinusoidalPositionalEmbedding.SinusoidalPositionalEmbedding(dim=self.hidden_size, init_size=DEFAULT_MAX_TARGET_POSITIONS, padding_idx=self.padding_idx, device=device)
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(SelfAttentionEncoderLayer(dim=self.hidden_size, kernel_size=kernel_size, num_heads=num_heads, dropout=dropout, norm_type=norm_type, padding_type=padding_type, act_type=act_type, device=device))
        if self.use_last_norm:
            if norm_type == "layer_norm":
                self.norm = nn.LayerNorm(self.hidden_size, device=device)
            else:
                self.norm = nn.BatchNorm1d(self.hidden_size, device=device)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, return_hidden: bool = False) -> torch.Tensor:
        padding_mask = torch.ones(x.shape[0], x.shape[1]).to(self.device) if padding_mask is None else padding_mask
        padding_mask = rearrange(padding_mask.float(), "b t -> t b ()")
        x_padding_mask = padding_mask.expand(-1, -1, x.shape[-1])
        if self.use_pos_embedding:
            positions = self.pos_embed_alpha * self.position_embedding(x[..., 0])
            x = x + positions
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = rearrange(x, "b t ... -> t b ...") * x_padding_mask
        hidden_layers = []
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask, attention_mask=attention_mask) * x_padding_mask
            if return_hidden:
                hidden_layers.append(x)
        if self.use_last_norm:
            if self.norm_type == "batch_norm":
                x = rearrange(x, "t b ... -> b ... t")
            x = self.norm(x)
            x = rearrange(x, "b ... t -> t b ...")
            x = x * x_padding_mask
        if return_hidden:
            x = torch.stack(hidden_layers, dim=0)
            x = rearrange(x, "l t b ... -> l b t ...")
        else:
            x = rearrange(x, "t b ... -> b t ...")
        return x