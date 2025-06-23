import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from Code_for_Experiment.Targeted_Training.dit_training_on_vae.music_dit2.modules.diffusion.dit.embeddings import TimestepEmbedding
from .dit_block import DiTBlock, FinalLayer
from ...embeddings import SinusoidalPosEmbedding, RotaryPosEmbedding

class DiT(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 num_heads: int,
                 output_dim: Optional[int] = None,
                 mlp_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 use_causality: bool = False,
                 use_cross_attention: bool = False,
                 use_rpr: bool = False,
                 context_dim: Optional[int] = None,
                 pos_embedding: Optional[str] = "RoPE",
                 max_position: int = 10000,
                 use_learned_variance: bool = True,
                 eps: float = 1e-6,
                 max_timestep_period: int = 10000,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.use_learned_variance = use_learned_variance
        self.input_dim = input_dim
        self.output_dim = output_dim or (input_dim * 2 if use_learned_variance else input_dim)
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.use_causality = use_causality
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            assert context_dim is not None, "Context dimension must be provided when using cross-attention."
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=eps)
        )
        self.timestep_embedding = TimestepEmbedding(hidden_dim=hidden_dim,
                                                    max_period=max_timestep_period)
        self.pos_embedding = pos_embedding
        if pos_embedding == "Sinusoidal":
            self.pos_embedding_layer = SinusoidalPosEmbedding(dim=hidden_dim,
                                                              max_position=max_position)
        elif pos_embedding == "RoPE":
            self.pos_embedding_layer = RotaryPosEmbedding(dim=hidden_dim,
                                                          max_position=max_position)
        else:
            assert pos_embedding is None or pos_embedding == "None", \
                "pos_embedding should be one of 'None', 'Sinusoidal' or 'RoPE'."
            self.pos_embedding_layer = None
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim=hidden_dim,
                     num_heads=num_heads,
                     mlp_dim=mlp_dim,
                     use_cross_attention=use_cross_attention,
                     context_dim=context_dim,
                     rope=self.pos_embedding_layer if pos_embedding == "RoPE" else None,
                     use_rpr=use_rpr,
                     max_position=max_position,
                     dropout=dropout,
                     eps=eps)
            for _ in range(num_layers)
        ])
        self.final_layer = FinalLayer(hidden_dim=hidden_dim,
                                      output_dim=self.output_dim,
                                      eps=eps)
        if device is not None:
            self.to(device)

    def initialize_weights(self):
        for block in self.blocks:
            block.initialize_weights()
        self.final_layer.initialize_weights()
        nn.init.xavier_uniform_(self.input_embedding[0].weight)

    def forward(self,
                x: torch.Tensor,
                *,
                timesteps: torch.Tensor,
                prompt: Optional[torch.Tensor] = None,
                positions: Optional[torch.Tensor] = None,
                condition: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None,
                context: Optional[torch.Tensor] = None,
                context_positions: Optional[torch.Tensor] = None,
                context_mask: Optional[torch.Tensor] = None) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert x.dim() == 3, "Input tensor must have 3 dimensions, as (batch_size, seq_len, input_dim)."
        assert x.size(2) == self.input_dim, \
            f"Input dimension {x.size(2)} must be the same as the model input dimension {self.input_dim}."
        sequence_len = x.size(1)
        if not self.use_cross_attention:
            assert context is None, "Context tensor should be None when not using cross-attention."
            assert context_positions is None, "Context positions tensor should be None when not using cross-attention."
            assert context_mask is None, "Context mask tensor should be None when not using cross-attention."
        if prompt is not None:
            assert prompt.dim() == 3, "Prompt tensor must have 3 dimensions, as (batch_size, prompt_len, input_dim)."
            assert prompt.size(2) == x.size(2), \
                f"Prompt dimension {prompt.size(2)} must be the same as the input dimension {x.size(2)}."
            assert prompt.size(0) == x.size(0), \
                f"Prompt batch size {prompt.size(0)} must be the same as the input batch size {x.size(0)}."
            prompt_len = prompt.size(1)
            x = torch.cat([prompt, x], dim=1)
        else:
            prompt_len = 0
        if positions is None:
            positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        else:
            assert positions.size() == (x.size(0), x.size(1)), \
                (f"Position shape {positions.size()} must be the same as the input shape,"
                 f"plus the prompt length {(x.size(0), x.size(1))}.")
        if self.pos_embedding == "Sinusoidal":
            pos_emb = self.pos_embedding_layer(positions)
            x = x + pos_emb
        if padding_mask is not None:
            assert padding_mask.size() == (x.size(0), sequence_len), \
                (f"Padding mask shape {padding_mask.size()} must be the same as the input shape "
                 f"{(x.size(0), x.size(1))}.")
            padding_mask = padding_mask.to(x.device)
            if prompt is not None:
                padding_mask = torch.cat([torch.ones(padding_mask.size(0), prompt_len, device=x.device),
                                          padding_mask], dim=1)
            x = x * padding_mask.unsqueeze(-1)
        if self.use_causality:
            causal_mask = torch.full((x.size(1), x.size(1)), float("-inf"), device=x.device)
            causal_mask = torch.triu(causal_mask, diagonal=1)
        else:
            causal_mask = None
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0).expand(x.size(0))
        assert timesteps.size() == (x.size(0), ), \
            f"Timestep shape {timesteps.size()} must be the same as the batch size {(x.size(0), )}."
        timestep_emb = self.timestep_embedding(timesteps).unsqueeze(1).expand(-1, x.size(1), -1).to(x.device)
        if condition is not None:
            assert condition.size() == (x.size(0), sequence_len, self.hidden_dim), \
                (f"Condition shape {condition.size()} must be the same as the input shape "
                 f"{(x.size(0), sequence_len, self.hidden_dim)}.")
            condition = condition.to(x.device)
            if prompt is not None:
                condition = torch.cat([torch.zeros(condition.size(0),
                                                   prompt_len,
                                                   condition.size(2),
                                                   device=x.device),
                                      condition], dim=1)
            condition = condition + timestep_emb
        else:
            condition = timestep_emb
        if context is not None:
            assert context.size() == (x.size(0), context.size(1), self.context_dim), \
                (f"Context shape {context.size()} must be the same as the input shape "
                 f"{(x.size(0), 'context_len', self.context_dim)}.")
            context = context.to(x.device)
            if context_positions is None:
                context_positions = (torch.arange(context.size(1), device=context.device).unsqueeze(0)
                                     .expand(context.size(0), -1))
            else:
                assert context_positions.size() == (context.size(0), context.size(1)), \
                    (f"Context position shape {context_positions.size()} must be the same as the context shape "
                     f"{(context.size(0), context.size(1))}.")
            if self.pos_embedding == "Sinusoidal":
                context_pos_emb = self.pos_embedding_layer(context_positions)
                context = context.to(x.device) + context_pos_emb
            if context_mask is not None:
                assert context_mask.size() == (context.size(0), context.size(1)), \
                    (f"Context padding mask shape {context_mask.size()} must be the same as the context shape "
                     f"{(context.size(0), context.size(1))}.")
                context = context * context_mask.unsqueeze(-1)
        x = self.input_embedding(x)
        for block in self.blocks:
            x = block(x,
                      condition=condition,
                      positions=positions,
                      padding_mask=padding_mask,
                      context=context,
                      context_positions=context_positions,
                      context_mask=context_mask,
                      causal_mask=causal_mask)
        x = self.final_layer(x, condition=condition)
        x = x[:, prompt_len:, :]
        if self.use_learned_variance:
            x_chunk = x.chunk(2, dim=-1)
            return x_chunk[0], x_chunk[1]
        else:
            return x

    def forward_with_cfg(self,
                         x: torch.Tensor,
                         *,
                         timesteps: torch.Tensor,
                         cfg_scale: float,
                         prompt: Optional[torch.Tensor] = None,
                         positions: Optional[torch.Tensor] = None,
                         condition: Optional[torch.Tensor] = None,
                         padding_mask: Optional[torch.Tensor] = None,
                         context: Optional[torch.Tensor] = None,
                         context_positions: Optional[torch.Tensor] = None,
                         context_mask: Optional[torch.Tensor] = None) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert condition is not None or context is not None, ("Condition or context tensor must be provided for "
                                                              "classifier-free guidance.")
        if self.use_learned_variance:
            out_cond, logvar_cond = self.forward(x,
                                                 timesteps=timesteps,
                                                 prompt=prompt,
                                                 positions=positions,
                                                 condition=condition,
                                                 padding_mask=padding_mask,
                                                 context=context,
                                                 context_positions=context_positions,
                                                 context_mask=context_mask)
            out_uncond, logvar_uncond = self.forward(x,
                                                     timesteps=timesteps,
                                                     prompt=prompt,
                                                     positions=positions,
                                                     padding_mask=padding_mask)
        else:
            out_cond = self.forward(x,
                                    timesteps=timesteps,
                                    prompt=prompt,
                                    positions=positions,
                                    condition=condition,
                                    padding_mask=padding_mask,
                                    context=context,
                                    context_positions=context_positions,
                                    context_mask=context_mask)
            out_uncond = self.forward(x,
                                      timesteps=timesteps,
                                      prompt=prompt,
                                      positions=positions,
                                      padding_mask=padding_mask)
            logvar_cond = logvar_uncond = None
        out = out_uncond + cfg_scale * (out_cond - out_uncond)
        if self.use_learned_variance:
            logvar = logvar_uncond + cfg_scale * (logvar_cond - logvar_uncond)
            return out, logvar
        else:
            return out