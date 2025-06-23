import torch
import torch.nn as nn
from typing import Optional
from .utils import modulate
from .activations import ApproxGELU
from ...transformers import Mlp, MultiHeadAttention
from Code_for_Experiment.Targeted_Training.dit_training_on_vae.music_dit2.modules.embeddings.rotary_pos_embedding import RotaryPosEmbedding

class DiTBlock(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 mlp_dim: Optional[int] = None,
                 use_cross_attention: bool = False,
                 context_dim: Optional[int] = None,
                 rope: Optional[RotaryPosEmbedding] = None,
                 use_rpr: bool = False,
                 max_position: int = 10000,
                 dropout: float = 0.0,
                 eps: float = 1e-6):
        super().__init__()
        mlp_dim = mlp_dim or hidden_dim * 4
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=eps)
        self.attn = MultiHeadAttention(input_dim=hidden_dim,
                                       num_heads=num_heads,
                                       qkv_bias=True,
                                       rope=rope,
                                       max_position=max_position,
                                       use_rpr=use_rpr)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=eps)
        self.mlp = Mlp(input_dim=hidden_dim,
                       hidden_dim=mlp_dim,
                       act_layer=ApproxGELU,
                       dropout=dropout)
        if use_cross_attention:
            self.cross_attn = MultiHeadAttention(input_dim=hidden_dim,
                                                 num_heads=num_heads,
                                                 qkv_bias=True,
                                                 context_dim=context_dim,
                                                 rope=rope,
                                                 max_position=max_position,
                                                 use_rpr=use_rpr)
            self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=eps)
        else:
            self.cross_attn = None
            self.norm3 = None
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6, bias=True)
        )

    def initialize_weights(self):
        self.attn.initialize_weights()
        self.mlp.initialize_weights()
        nn.init.xavier_uniform_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
        if self.cross_attn is not None:
            self.cross_attn.initialize_weights()

    def forward(self,
                x: torch.Tensor,
                *,
                condition: torch.Tensor,
                positions: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None,
                context: Optional[torch.Tensor] = None,
                context_positions: Optional[torch.Tensor] = None,
                context_mask: Optional[torch.Tensor] = None,
                causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(self.adaLN_modulation(condition), chunks=6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa),
                                     positions=positions,
                                     padding_mask=padding_mask,
                                     causal_mask=causal_mask)
        if self.cross_attn is not None and context is not None:
            x = x + self.cross_attn(self.norm3(x),
                                    positions=positions,
                                    padding_mask=padding_mask,
                                    context=context,
                                    context_positions=context_positions,
                                    context_mask=context_mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 output_dim: int,
                 eps: float = 1e-6):
        super().__init__()
        self.output_dim = output_dim
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=eps)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2, bias=True)
        )

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.xavier_uniform_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self,
                x: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(condition).chunk(chunks=2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x