import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as distributed
from torch.optim import Optimizer
from einops import rearrange, repeat
from typing import Optional
from functools import partial
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger
from .codebooks import EuclideanCodebook, CosineSimCodebook
from .utils import (gumbel_sample, identity, pack_one, unpack_one, orthogonal_loss_fn)

logger = get_logger(__name__)

class VectorQuantization(nn.Module):
    def __init__(self, 
                 dim: int, 
                 codebook_size: int, 
                 codebook_dim: Optional[int] = None, 
                 num_heads: int = 1, 
                 separate_codebook_per_head: bool = False, 
                 decay: float = 0.8, 
                 eps: float = 1e-7, 
                 kmeans_init: bool = False, 
                 kmeans_iters: bool = 100, 
                 sync_kmeans: bool = True, 
                 similarity: str = 'cosine', 
                 threshold_ema_dead_code: int = 2, 
                 channel_last: bool = True, 
                 commitment_weight: float = 1.0, 
                 commitment_use_cross_entropy_loss: bool = False, 
                 orthogonal_reg_weight: float = 0.0, 
                 orthogonal_reg_active_codes_only: bool = False, 
                 orthogonal_reg_max_codes: Optional[int] = None, 
                 stochastic_sample_codes: bool = False, 
                 sample_codebook_temp: float = 1.0, 
                 straight_through: bool = False, 
                 reinmax: bool = False, 
                 sync_codebook: Optional[bool] = None, 
                 sync_affine_param: bool = False, 
                 ema_update: bool = True, 
                 learnable_codebook: bool = False, 
                 in_place_codebook_optimizer: Optional[Optimizer.__class__] = None, 
                 learning_rate: float = 1e-3, 
                 affine_param: bool = False, 
                 affine_param_batch_decay: float = 0.99, 
                 affine_param_codebook_decay: float = 0.9, 
                 sync_update_v: float = 0.0):
        super(VectorQuantization, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.separate_codebook_per_head = separate_codebook_per_head
        codebook_dim = codebook_dim if codebook_dim is not None else dim
        codebook_input_dim = codebook_dim * num_heads
        requires_projection = (codebook_input_dim != dim)
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else identity
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else identity
        self.eps = eps
        self.commitment_weight = commitment_weight
        self.commitment_use_cross_entropy_loss = commitment_use_cross_entropy_loss
        self.learnable_codebook = learnable_codebook
        has_codebook_orthogonal_loss = orthogonal_reg_weight > 0
        self.has_codebook_orthogonal_loss = has_codebook_orthogonal_loss
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes: Optional[int] = orthogonal_reg_max_codes
        assert not (ema_update and learnable_codebook), 'Learnable codebook not compatible with EMA update.'
        assert 0 <= sync_update_v <= 1.
        assert not (sync_update_v > 0. and not learnable_codebook), "Learnable codebook must be turned on for " \
                                                                    "synchronous update rule (21)."
        self.sync_update_v = sync_update_v
        assert similarity in ('cosine', 'euclidean'), "similarity measure must be either cosine or euclidean"
        codebook_class = EuclideanCodebook if similarity == 'euclidean' else CosineSimCodebook
        gumbel_sample_fn = partial(
            gumbel_sample,
            stochastic=stochastic_sample_codes,
            reinmax=reinmax,
            straight_through=straight_through
        )
        if sync_codebook is None:
            sync_codebook = distributed.is_initialized() and distributed.get_world_size() > 1
        codebook_kwargs = dict(
            dim=codebook_dim,
            num_codebooks=num_heads if separate_codebook_per_head else 1,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            sync_kmeans=sync_kmeans,
            decay=decay,
            eps=eps,
            threshold_ema_dead_code=threshold_ema_dead_code,
            use_ddp=sync_codebook,
            learnable_codebook=has_codebook_orthogonal_loss or learnable_codebook,
            sample_codebook_temp=sample_codebook_temp,
            gumbel_sample=gumbel_sample_fn,
            ema_update=ema_update
        )
        if affine_param:
            assert similarity == "euclidean", "affine param is only compatible with euclidean codebook"
            codebook_kwargs = dict(
                **codebook_kwargs,
                affine_param=True,
                sync_affine_param=sync_affine_param,
                affine_param_batch_decay=affine_param_batch_decay,
                affine_param_codebook_decay=affine_param_codebook_decay,
            )
        self.codebook_obj = codebook_class(**codebook_kwargs)
        if in_place_codebook_optimizer is not None:
            assert learnable_codebook, 'In-place codebook optimizer requires learnable codebook.'
            assert not ema_update, 'In-place codebook optimizer requires EMA update to be off.'
            self.in_place_codebook_optimizer = in_place_codebook_optimizer(self.codebook_obj.parameters(), lr=learning_rate)
        else:
            self.in_place_codebook_optimizer = None
        self.codebook_size = codebook_size
        self.channel_last = channel_last

    @property
    def codebook(self) -> torch.Tensor:
        codebook = self.codebook_obj.embed
        if self.separate_codebook_per_head:
            return codebook
        return rearrange(codebook, '1 ... -> ...')

    @codebook.setter
    def codebook(self, codes: torch.Tensor):
        if not self.separate_codebook_per_head:
            codes = rearrange(codes, '... -> 1 ...')
        self.codebook_obj.embed.copy_(codes)

    def get_codes_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        codebook = self.codebook
        is_multiheaded = codebook.dim() > 2
        if not is_multiheaded:
            codes = codebook[indices]
            return rearrange(codes, '... h d -> ... (h d)')
        indices, ps = pack_one(indices, 'b * h')
        indices = rearrange(indices, 'b n h -> b h n')
        indices = repeat(indices, 'b h n -> b h n d', d=codebook.shape[-1])
        codebook = repeat(codebook, 'h n d -> b h n d', b=indices.shape[0])
        codes = codebook.gather(2, indices)
        codes = rearrange(codes, 'b h n d -> b n (h d)')
        codes = unpack_one(codes, ps, 'b * d')
        return codes

    def get_output_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        codes = self.get_codes_from_indices(indices)
        return self.project_out(codes)

    def forward(self, 
                x: torch.Tensor, 
                indices: Optional[torch.Tensor] = None, 
                mask: Optional[torch.Tensor] = None, 
                sample_codebook_temp: Optional[float] = None, 
                freeze_codebook: bool = False) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        orig_input = x
        assert x.dim() == 3 or x.dim() == 4, 'input must be 3 or 4-dimensional'
        shape = x.shape
        device = x.device
        is_multiheaded = self.num_heads > 1
        return_loss = (indices is not None) and self.training
        need_transpose = not self.channel_last
        should_inplace_optimize = (self.in_place_codebook_optimizer is not None)
        if need_transpose:
            assert x.shape[-2] == self.dim, \
                f'Input tensor must have the same dimension ({x.shape[-2]}) as the module (self.dim).'
        else:
            assert x.shape[-1] == self.dim, \
                f'Input tensor must have the same dimension ({x.shape[-1]}) as the module (self.dim).'
        if need_transpose:
            x = rearrange(x, 'b d n -> b n d')
        x = self.project_in(x)
        if is_multiheaded:
            ein_rhs_eq = 'h b n d' if self.separate_codebook_per_head else '1 (b h) n d'
            x = rearrange(x, f'b n (h d) -> {ein_rhs_eq}', h=self.num_heads)
        x = self.codebook_obj.transform_input(x)
        codebook_forward_kwargs = dict(
            sample_codebook_temp=sample_codebook_temp,
            mask=mask,
            freeze_codebook=freeze_codebook
        )
        quantize, embed_ind, distances = self.codebook_obj(x, **codebook_forward_kwargs)
        if should_inplace_optimize and self.training and not freeze_codebook:
            if mask is not None:
                loss = F.mse_loss(quantize, x.detach(), reduction='none')
                loss_mask = mask
                if is_multiheaded:
                    loss_mask = repeat(mask, 'b n -> c (b h) n', c=loss.shape[0], h=loss.shape[1] // mask.shape[0])
                loss = loss[loss_mask].mean()
            else:
                loss = F.mse_loss(quantize, x.detach())
            loss.backward()
            self.in_place_codebook_optimizer.step()
            self.in_place_codebook_optimizer.zero_grad()
            quantize, embed_ind, distances = self.codebook_obj(x, **codebook_forward_kwargs)
        if self.training:
            maybe_detach = torch.detach if not self.learnable_codebook or freeze_codebook else identity
            commit_quantize = maybe_detach(quantize)
            quantize = x + (quantize - x).detach()
            if self.sync_update_v > 0.:
                quantize = quantize + self.sync_update_v * (quantize - quantize.detach())
        else:
            commit_quantize = None

        def calculate_ce_loss(codes):
            if not is_multiheaded:
                dist_einops_eq = '1 b n l -> b l n'
            elif self.separate_codebook_per_head:
                dist_einops_eq = 'c b n l -> b l n c'
            else:
                dist_einops_eq = '1 (b h) n l -> b l n h'
            ce_loss = F.cross_entropy(
                rearrange(distances, dist_einops_eq, b=shape[0]),
                codes,
                ignore_index=-1
            )
            return ce_loss

        if return_loss:
            return quantize, calculate_ce_loss(indices)
        if is_multiheaded:
            if self.separate_codebook_per_head:
                embed_ind = rearrange(embed_ind, 'h b n -> b n h', h=self.num_heads)
            else:
                embed_ind = rearrange(embed_ind, '1 (b h) n -> b n h', h=self.num_heads)
        loss = torch.tensor([0.], device=device, requires_grad=self.training)
        if self.training:
            if self.commitment_weight > 0:
                if self.commitment_use_cross_entropy_loss:
                    if mask is not None:
                        ce_loss_mask = mask
                    if is_multiheaded:
                        ce_loss_mask = repeat(ce_loss_mask, 'b n -> b n h', h=self.num_heads)
                    embed_ind.masked_fill_(~ce_loss_mask, -1)
                    commit_loss = calculate_ce_loss(embed_ind)
                else:
                    if mask is not None:
                        commit_loss = F.mse_loss(commit_quantize, x, reduction='none')
                        loss_mask = mask
                        if is_multiheaded:
                            loss_mask = repeat(loss_mask, 'b n -> c (b h) n', c=commit_loss.shape[0], h=commit_loss.shape[1] // mask.shape[0])
                        commit_loss = commit_loss[loss_mask].mean()
                    else:
                        commit_loss = F.mse_loss(commit_quantize, x)
                loss = loss + commit_loss * self.commitment_weight
            if self.has_codebook_orthogonal_loss:
                codebook = self.codebook_obj.embed
                if self.orthogonal_reg_active_codes_only:
                    assert not (is_multiheaded and self.separate_codebook_per_head), \
                        'Cannot apply orthogonal regularization only to active codes when the VQ is multi-headed.'
                    unique_code_ids = torch.unique(embed_ind)
                    codebook = codebook[:, unique_code_ids]
                    num_codes = codebook.shape[-2]
                    if (self.orthogonal_reg_max_codes is not None and num_codes > self.orthogonal_reg_max_codes):
                        rand_ids = torch.randperm(num_codes, device=device)[:self.orthogonal_reg_max_codes]
                        codebook = codebook[:, rand_ids]
                        orthogonal_reg_loss = orthogonal_loss_fn(codebook)
                    loss = loss + orthogonal_reg_loss * self.orthogonal_reg_weight
        if is_multiheaded:
            if self.separate_codebook_per_head:
                quantize = rearrange(quantize, 'h b n d -> b n (h d)', h=self.num_heads)
            else:
                quantize = rearrange(quantize, '1 (b h) n d -> b n (h d)', h=self.num_heads)
        else:
            if quantize.dim() == 4:
                quantize = rearrange(quantize, '1 b n d -> b n d')
        quantize = self.project_out(quantize)
        if need_transpose:
            quantize = rearrange(quantize, 'b n d -> b d n')
        if mask is not None:
            quantize = torch.where(rearrange(mask, '... -> ... 1'), quantize, orig_input)
        return quantize, embed_ind, loss