import torch
import torch.nn as nn
import torch.nn.functional as F
import einx
import random
from IPython import embed
from torch.optim import Optimizer
from typing import Optional
from functools import partial
from einops import rearrange, reduce
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.modules.rvq.vq import VectorQuantization
from .utils import identity, pack, unpack, round_up_multiple

class ResidualVectorQuantization(nn.Module):
    def __init__(self, *, dim: int, num_quantizers: int, codebook_size: int, codebook_dim: Optional[int] = None, decay: float = 0.8, eps: float = 1e-7, kmeans_init: bool = False, kmeans_iters: bool = 100, sync_kmeans: bool = True, similarity: str = 'cosine', threshold_ema_dead_code: int = 2, channel_last: bool = True, commitment_weight: float = 1.0, commitment_use_cross_entropy_loss: bool = False, orthogonal_reg_weight: float = 0.0, orthogonal_reg_active_codes_only: bool = False, orthogonal_reg_max_codes: Optional[int] = None, stochastic_sample_codes: bool = False, sample_codebook_temp: float = 1.0, straight_through: bool = False, reinmax: bool = False, sync_codebook: Optional[bool] = None, sync_affine_param: bool = False, ema_update: bool = True, learnable_codebook: bool = False, in_place_codebook_optimizer: Optional[Optimizer.__class__] = None, learning_rate: float = 1e-3, affine_param: bool = False, affine_param_batch_decay: float = 0.99, affine_param_codebook_decay: float = 0.9, sync_update_v: float = 0.0, quantize_dropout: bool = False, quantize_dropout_cutoff_index: int = 0, quantize_dropout_multiple_of: int = 1, shared_codebook: bool = False):
        super(ResidualVectorQuantization, self).__init__()
        codebook_dim = codebook_dim if codebook_dim is not None else dim
        requires_projection = (codebook_dim != dim)
        self.project_in = nn.Linear(dim, codebook_dim) if requires_projection else identity
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection else identity
        self.num_quantizers = num_quantizers
        self.dim = dim
        self.channel_last = channel_last
        self.layers = nn.ModuleList([
            VectorQuantization(dim=codebook_dim,
                               codebook_size=codebook_size,
                               codebook_dim=codebook_dim,
                               num_heads=1,
                               separate_codebook_per_head=False,
                               decay=decay,
                               eps=eps,
                               kmeans_init=kmeans_init,
                               kmeans_iters=kmeans_iters,
                               sync_kmeans=sync_kmeans,
                               similarity=similarity,
                               threshold_ema_dead_code=threshold_ema_dead_code,
                               channel_last=channel_last,
                               commitment_weight=commitment_weight,
                               commitment_use_cross_entropy_loss=commitment_use_cross_entropy_loss,
                               orthogonal_reg_weight=orthogonal_reg_weight,
                               orthogonal_reg_active_codes_only=orthogonal_reg_active_codes_only,
                               orthogonal_reg_max_codes=orthogonal_reg_max_codes,
                               stochastic_sample_codes=stochastic_sample_codes,
                               sample_codebook_temp=sample_codebook_temp,
                               straight_through=straight_through,
                               reinmax=reinmax,
                               sync_codebook=sync_codebook,
                               sync_affine_param=sync_affine_param,
                               ema_update=ema_update,
                               learnable_codebook=learnable_codebook,
                               in_place_codebook_optimizer=in_place_codebook_optimizer,
                               learning_rate=learning_rate,
                               affine_param=affine_param,
                               affine_param_batch_decay=affine_param_batch_decay,
                               affine_param_codebook_decay=affine_param_codebook_decay,
                               sync_update_v=sync_update_v)
            for _ in range(num_quantizers)
        ])
        self.quantize_dropout = quantize_dropout and num_quantizers > 1
        assert quantize_dropout_cutoff_index >= 0, 'Quantize dropout cutoff index must be non-negative.'
        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of
        if not shared_codebook:
            return
        first_vq, *rest_vq = self.layers
        codebook = first_vq.codebook_obj
        for vq in rest_vq:
            vq.codebook_obj = codebook

    @property
    def codebooks(self) -> torch.Tensor:
        codebooks = [layer.codebook for layer in self.layers]
        codebooks = torch.stack(codebooks, dim=0)
        return codebooks

    def get_codes_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        batch, quantize_dim = indices.shape[0], indices.shape[-1]
        indices, ps = pack([indices], 'b * q')
        if quantize_dim < self.num_quantizers:
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value=-1)
        mask = indices == -1.
        indices = indices.masked_fill(mask, 0)
        all_codes = einx.get_at('q [c] d, b n q -> q b n d', self.codebooks, indices)
        all_codes = all_codes.masked_fill(rearrange(mask, 'b n q -> q b n 1'), 0.)
        all_codes, = unpack(all_codes, ps, 'q b * d')
        return all_codes

    def get_output_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        codes = self.get_codes_from_indices(indices)
        codes_summed = reduce(codes, 'q ... -> ...', 'sum')
        return self.project_out(codes_summed)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, indices: Optional[torch.Tensor] = None, num_quantizers: Optional[int] = None, return_all_codes: bool = False, sample_codebook_temp: Optional[float] = None, freeze_codebook: Optional[bool] = False, rand_quantize_dropout_fixed_seed: Optional[int] = None) -> dict:
        return_loss = (indices is not None) and self.training
        need_transpose = not self.channel_last
        assert x.dim() == 3 or x.dim() == 4, 'Input tensor must be 3D or 4D.'
        if need_transpose:
            assert x.shape[-2] == self.dim, f'Input tensor must have the same dimension ({x.shape[-2]}) as the module (self.dim).'
        else:
            assert x.shape[-1] == self.dim, f'Input tensor must have the same dimension ({x.shape[-1]}) as the module (self.dim).'
        x = self.project_in(x)
        quantized_out = 0.0
        residual = x
        all_losses = []
        all_indices = []
        ce_losses = []
        if return_loss:
            assert not torch.any(torch.Tensor(indices < 0)), ("Some of the residual vq indices were dropped out. please use indices derived when " "the module is in eval mode to derive cross entropy loss")
        should_quantize_dropout = self.training and self.quantize_dropout and not return_loss
        if should_quantize_dropout:
            rand = random.Random(rand_quantize_dropout_fixed_seed) if rand_quantize_dropout_fixed_seed is not None else random
            rand_quantize_dropout_index = rand.randrange(self.quantize_dropout_cutoff_index, self.num_quantizers)
            if self.quantize_dropout_multiple_of != 1:
                rand_quantize_dropout_index = round_up_multiple(rand_quantize_dropout_index + 1, self.quantize_dropout_multiple_of) - 1
            null_indices_shape = tuple(x.shape[:2])
            null_indices = torch.full(null_indices_shape, -1., device=x.device, dtype=torch.long)
            null_loss = torch.full((1,), 0., device=x.device, dtype=x.dtype)
        else:
            rand_quantize_dropout_index = None
            null_indices = None
            null_loss = None
        for quantizer_index, layer in enumerate(self.layers):
            if num_quantizers is not None and quantizer_index >= num_quantizers:
                break
            if should_quantize_dropout and quantizer_index > rand_quantize_dropout_index:
                all_indices.append(null_indices)
                all_losses.append(null_loss)
                continue
            layer_indices = None
            if return_loss:
                layer_indices = indices[..., quantizer_index]
                quantized, *rest = layer(residual, mask=mask, indices=layer_indices, sample_codebook_temp=sample_codebook_temp, freeze_codebook=freeze_codebook)
                residual = residual - quantized.detach()
                quantized_out = quantized_out + quantized
                if return_loss:
                    ce_loss = rest[0]
                    ce_losses.append(ce_loss)
                    continue
                embed_indices, loss = rest
                all_indices.append(embed_indices)
                all_losses.append(loss)
        quantized_out = self.project_out(quantized_out)
        if return_loss:
            return {"quantized_out": quantized_out, "loss": sum(ce_losses)}
        all_losses, all_indices = map(partial(torch.stack, dim=-1), (all_losses, all_indices))
        ret = {"quantized_out": quantized_out, "all_indices": all_indices, "all_losses": all_losses}
        if return_all_codes:
            all_codes = self.get_codes_from_indices(all_indices)
            ret["all_codes"] = all_codes
        return ret