from abc import ABC, abstractmethod
import torch.nn as nn
import torch.distributed as distributed
from torch.cuda.amp import autocast
from einops import rearrange, reduce, repeat
from functools import partial
from typing import Optional, Callable
from .utils import (noop, uniform_init, kmeans, identity, gumbel_sample as gumbel_sample_0, cdist, l2norm, pack_one, unpack_one, batched_sample_vectors, sample_vectors_distributed, ema_inplace_, laplace_smoothing, batched_embedding)
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger
logger = get_logger(__name__)

class CodebookBase(nn.Module, ABC):
    def __init__(self, codebook_size: int, num_codebooks: int = 1, kmeans_init: bool = False, kmeans_iters: int = 100, sync_kmeans: bool = True, decay: float = 0.8, eps: float = 1e-7, threshold_ema_dead_code: int = 2, reset_cluster_size: Optional[int] = None, use_ddp: bool = False, gumbel_sample: Callable = gumbel_sample_0, sample_codebook_temp=1.0, use_cosine_sim: bool = False, ema_update=True):
        super(CodebookBase, self).__init__()
        self.transform_input = identity
        self.decay = decay
        self.use_cosine_sim = use_cosine_sim
        self.ema_update = ema_update
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.reset_cluster_size = reset_cluster_size if reset_cluster_size is not None else threshold_ema_dead_code
        assert callable(gumbel_sample)
        self.gumbel_sample = gumbel_sample
        self.sample_codebook_temp = sample_codebook_temp
        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop
        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(num_codebooks, codebook_size))

    @torch.jit.ignore
    def init_embed_(self, data: torch.Tensor, mask: torch.Tensor = None):
        if self.initted:
            return
        if mask is not None:
            c = data.shape[0]
            data = rearrange(data[mask], '(c n) d -> c n d', c=c)
        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters, use_cosine_sim=self.use_cosine_sim, sample_fn=self.sample_fn, all_reduce_fn=self.kmeans_all_reduce_fn)
        embed_sum = embed * rearrange(cluster_size, '... -> ... 1')
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed_sum)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace_(self, batch_samples: torch.Tensor, batch_mask: torch.Tensor):
        if self.use_cosine_sim:
            batch_samples = l2norm(batch_samples)
        for ind, (samples, mask) in enumerate(zip(batch_samples.unbind(dim=0), batch_mask.unbind(dim=0))):
            if not torch.any(mask):
                continue
            sampled = self.sample_fn(rearrange(samples, '... -> 1 ...'), mask.sum().item())
            sampled = rearrange(sampled, '1 ... -> ...')
            self.embed.data[ind][mask] = sampled
            self.cluster_size.data[ind][mask] = self.reset_cluster_size
            self.embed_avg.data[ind][mask] = sampled * self.reset_cluster_size

    def expire_codes_(self, batch_samples: torch.Tensor):
        if self.threshold_ema_dead_code == 0:
            return
        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return
        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        self.replace_(batch_samples, batch_mask=expired_codes)

    def preprocess(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        x = x.float()
        assert x.dim() == 3 or x.dim() == 4, 'input tensor must be 3D or 4D.'
        if x.dim() < 4:
            x = rearrange(x, '... -> 1 ...')
        flatten, ps = pack_one(x, 'h * d')
        if mask is not None:
            mask = repeat(mask, 'b n -> c (b h n)', c=flatten.shape[0], h=flatten.shape[-2] // (mask.shape[0] * mask.shape[1]))
        self.init_embed_(flatten, mask=mask)
        return x, mask, flatten, ps

    def do_sample_and_update_embed(self, x: torch.Tensor, orig_x_dim: int, dist: torch.Tensor, embed: torch.Tensor, flatten: torch.Tensor, ps: torch.Tensor, mask: Optional[torch.Tensor] = None, sample_codebook_temp: Optional[float] = None, freeze_codebook: bool = False) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        sample_codebook_temp = sample_codebook_temp if sample_codebook_temp is not None else self.sample_codebook_temp
        embed_ind, embed_onehot = self.gumbel_sample(dist, dim=-1, temperature=sample_codebook_temp, training=self.training)
        embed_ind = unpack_one(embed_ind, ps, 'h *')
        if self.training:
            unpacked_onehot = unpack_one(embed_onehot, ps, 'h * c')
            quantize = torch.einsum('h b n c, h c d -> h b n d', unpacked_onehot, embed)
        else:
            quantize = batched_embedding(embed_ind, embed)
        if self.training and self.ema_update and not freeze_codebook:
            if mask is not None:
                embed_onehot[~mask] = 0.
            cluster_size = embed_onehot.sum(dim=1)
            self.all_reduce_fn(cluster_size)
            ema_inplace_(self.cluster_size.data, cluster_size, self.decay)
            embed_sum = torch.einsum('h n d, h n c -> h c d', flatten, embed_onehot)
            embed_sum = embed_sum.contiguous()
            self.all_reduce_fn(embed_sum)
            ema_inplace_(self.embed_avg.data, embed_sum, self.decay)
            cluster_size = (laplace_smoothing(self.cluster_size, self.codebook_size, eps=self.eps) * self.cluster_size.sum(dim=-1, keepdim=True))
            embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
            if self.use_cosine_sim:
                embed_normalized = l2norm(embed_normalized)
            self.embed.data.copy_(embed_normalized)
            self.expire_codes_(x)
        if orig_x_dim < 4:
            quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))
        dist = unpack_one(dist, ps, 'h * d')
        return quantize, embed_ind, dist

    @abstractmethod
    def forward(self, x: torch.Tensor, sample_codebook_temp: Optional[float] = None, mask: Optional[torch.Tensor] = None, freeze_codebook: bool = False) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        ...

class EuclideanCodebook(CodebookBase):
    def __init__(self, dim: int, codebook_size: int, num_codebooks: int = 1, kmeans_init: bool = False, kmeans_iters: int = 100, sync_kmeans: bool = True, decay: float = 0.8, eps: float = 1e-7, threshold_ema_dead_code: int = 2, reset_cluster_size: Optional[int] = None, use_ddp: bool = False, learnable_codebook: bool = False, gumbel_sample: Callable = gumbel_sample_0, sample_codebook_temp=1.0, ema_update=True, affine_params=False, sync_affine_params=False, affine_param_batch_decay=0.99, affine_param_codebook_decay=0.9):
        super(EuclideanCodebook, self).__init__(codebook_size=codebook_size, num_codebooks=num_codebooks, kmeans_init=kmeans_init, kmeans_iters=kmeans_iters, sync_kmeans=sync_kmeans, decay=decay, eps=eps, threshold_ema_dead_code=threshold_ema_dead_code, reset_cluster_size=reset_cluster_size, use_ddp=use_ddp, gumbel_sample=gumbel_sample, sample_codebook_temp=sample_codebook_temp, use_cosine_sim=False, ema_update=ema_update)
        init_fn = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(num_codebooks, codebook_size, dim)
        assert not (use_ddp and num_codebooks > 1 and kmeans_init), 'kmeans init is not compatible with multiple codebooks in distributed environment for now'
        self.register_buffer('embed_avg', embed.clone())
        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)
        self.affine_params = affine_params
        self.sync_affine_params = sync_affine_params
        if not affine_params:
            return
        self.affine_param_batch_decay = affine_param_batch_decay
        self.affine_param_codebook_decay = affine_param_codebook_decay
        self.register_buffer('batch_mean', None)
        self.register_buffer('batch_variance', None)
        self.register_buffer('codebook_mean_needs_init', torch.Tensor([True]))
        self.register_buffer('codebook_mean', torch.empty(num_codebooks, 1, dim))
        self.register_buffer('codebook_variance_needs_init', torch.Tensor([True]))
        self.register_buffer('codebook_variance', torch.empty(num_codebooks, 1, dim))

    @torch.jit.ignore
    def update_with_decay_(self, buffer_name: str, new_value: torch.Tensor, decay: float):
        old_value = getattr(self, buffer_name)
        needs_init = getattr(self, buffer_name + "_needs_init", False)
        if needs_init:
            self.register_buffer(buffer_name + "_needs_init", torch.Tensor([False]))
        if old_value is None or needs_init:
            self.register_buffer(buffer_name, new_value.detach())
            return
        value = old_value * decay + new_value.detach() * (1 - decay)
        self.register_buffer(buffer_name, value)

    @torch.jit.ignore
    def update_affine_(self, data: torch.Tensor, embed: torch.Tensor, mask: torch.Tensor = None):
        assert self.affine_params, "Affine parameters are not enabled."
        var_fn = partial(torch.var, unbiased=False)
        embed = rearrange(embed, 'h ... d -> h (...) d')
        if self.training:
            self.update_with_decay_('codebook_mean', reduce(embed, 'h n d -> h 1 d', 'mean'), self.affine_param_codebook_decay)
            self.update_with_decay_('codebook_variance', reduce(embed, 'h n d -> h 1 d', var_fn), self.affine_param_codebook_decay)
        data = rearrange(data, 'h ... d -> h (...) d')
        if mask is not None:
            c = data.shape[0]
            data = rearrange(data[mask], '(c n) d -> c n d', c=c)
        if not self.sync_affine_params:
            self.update_with_decay_('batch_mean', reduce(data, 'h n d -> h 1 d', 'mean'), self.affine_param_batch_decay)
            self.update_with_decay_('batch_variance', reduce(data, 'h n d -> h 1 d', var_fn), self.affine_param_batch_decay)
            return
        num_vectors, device, dtype = data.shape[-2], data.device, data.dtype
        num_vectors = torch.tensor([num_vectors], device=device, dtype=dtype)
        self.all_reduce_fn(num_vectors)
        batch_sum = reduce(data, 'h n d -> h 1 d', 'sum')
        self.all_reduce_fn(batch_sum)
        batch_mean = batch_sum / num_vectors
        self.update_with_decay_('batch_mean', batch_mean, decay=self.affine_param_batch_decay)
        variance_numer = reduce((data - batch_mean) ** 2, 'h n d -> h 1 d', 'sum')
        self.all_reduce_fn(variance_numer)
        batch_variance = variance_numer / num_vectors
        self.update_with_decay_('batch_variance', batch_variance, decay=self.affine_param_batch_decay)

    @autocast(enabled=False)
    def forward(self, x: torch.Tensor, sample_codebook_temp: float = None, mask: torch.Tensor = None, freeze_codebook: bool = False) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        sample_codebook_temp = sample_codebook_temp if sample_codebook_temp is not None else self.sample_codebook_temp
        orig_x_dim = x.dim()
        x, mask, flatten, ps = self.preprocess(x, mask=mask)
        if self.affine_params:
            self.update_affine_(flatten, self.embed, mask=mask)
        embed = self.embed if self.learnable_codebook else self.embed.detach()
        if self.affine_params:
            codebook_std = self.codebook_variance.clamp(min=1e-7).sqrt()
            batch_std = self.batch_variance.clamp(min=1e-7).sqrt()
            embed = (embed - self.codebook_mean) * (batch_std / codebook_std) + self.batch_mean
            if self.training and self.ema_update and not freeze_codebook:
                flatten = (flatten - self.batch_mean) * (codebook_std / batch_std) + self.codebook_mean
        dist = -cdist(flatten, embed)
        return self.do_sample_and_update_embed(x, orig_x_dim=orig_x_dim, dist=dist, embed=embed, flatten=flatten, ps=ps, mask=mask, sample_codebook_temp=sample_codebook_temp, freeze_codebook=freeze_codebook)

class CosineSimCodebook(CodebookBase):
    def __init__(self, dim: int, codebook_size: int, num_codebooks: int = 1, kmeans_init: bool = False, kmeans_iters: int = 100, sync_kmeans: bool = True, decay: float = 0.8, eps: float = 1e-7, threshold_ema_dead_code: int = 2, reset_cluster_size: Optional[int] = None, use_ddp: bool = False, learnable_codebook: bool = False, gumbel_sample: Callable = gumbel_sample_0, sample_codebook_temp: float = 1.0, ema_update: bool = True):
        super(CosineSimCodebook, self).__init__(codebook_size=codebook_size, num_codebooks=num_codebooks, kmeans_init=kmeans_init, kmeans_iters=kmeans_iters, sync_kmeans=sync_kmeans, decay=decay, eps=eps, threshold_ema_dead_code=threshold_ema_dead_code, reset_cluster_size=reset_cluster_size, use_ddp=use_ddp, gumbel_sample=gumbel_sample, sample_codebook_temp=sample_codebook_temp, use_cosine_sim=True, ema_update=ema_update)
        if not kmeans_init:
            embed = l2norm(uniform_init(num_codebooks, codebook_size, dim))
        else:
            embed = torch.zeros(num_codebooks, codebook_size, dim)
        self.register_buffer('embed_avg', embed.clone())
        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)

    @autocast(enabled=False)
    def forward(self, x: torch.Tensor, sample_codebook_temp: Optional[float] = None, mask: Optional[torch.Tensor] = None, freeze_codebook: bool = False) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        sample_codebook_temp = sample_codebook_temp if sample_codebook_temp is not None else self.sample_codebook_temp
        orig_x_dim = x.dim()
        x, mask, flatten, ps = self.preprocess(x, mask=mask)
        embed = self.embed if self.learnable_codebook else self.embed.detach()
        dist = torch.einsum('h n d, h c d -> h n c', flatten, embed)
        return self.do_sample_and_update_embed(x, orig_x_dim=orig_x_dim, dist=dist, embed=embed, flatten=flatten, ps=ps, mask=mask, sample_codebook_temp=sample_codebook_temp, freeze_codebook=freeze_codebook)