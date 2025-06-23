import math
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from typing import Callable, Iterable
from einops import rearrange, repeat, reduce, pack, unpack
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.misc import safe_softmax
def ema_inplace_(old: torch.Tensor, new: torch.Tensor, decay: float = 0.99):
    is_mps = str(old.device).startswith('mps')
    if not is_mps:
        old.lerp_(new, 1 - decay)
    else:
        old.mul_(decay).add_(new * (1 - decay))
def uniform_init(*shape) -> torch.Tensor:
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t
def identity(t: any) -> any:
    return t
def log(t: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return torch.log(t.clamp(min=eps))
def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(1)
    return -log(-log(noise))
def gumbel_sample(logits: torch.Tensor, temperature: float = 1.0, stochastic: bool = False, straight_through: bool = False, reinmax: bool = False, dim: int = -1, training: bool = True) -> (torch.Tensor, torch.Tensor):
    dtype, size = logits.dtype, logits.shape[dim]
    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits
    ind = sampling_logits.argmax(dim=dim)
    one_hot = F.one_hot(ind, size).type(dtype)
    assert not (reinmax and not straight_through), 'reinmax can only be turned on if using straight through gumbel softmax'
    if not straight_through or temperature <= 0. or not training:
        return ind, one_hot
    if reinmax:
        p0 = safe_softmax(logits, dim=dim)
        p1 = (one_hot + safe_softmax(logits / temperature, dim=dim)) / 2
        p1 = safe_softmax((log(p1 + 1e-7) - logits).detach() + logits, dim=1)
        p2 = 2 * p1 - 0.5 * p0
        one_hot = p2 - p2.detach() + one_hot
    else:
        p1 = safe_softmax(logits / temperature, dim=dim)
        one_hot = one_hot + p1 - p1.detach()
    return ind, one_hot
def laplace_smoothing(x: torch.Tensor, n_categories: int, *, eps: float = 1e-7, dim: int = -1) -> torch.Tensor:
    denom = x.sum(dim=dim, keepdim=True)
    return (x + eps) / (denom + n_categories * eps)
def sample_vectors(samples: torch.Tensor, num: int) -> torch.Tensor:
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)
    return samples[indices]
def noop(*args, **kwargs):
    pass
def l2norm(t: torch.Tensor) -> torch.Tensor:
    return F.normalize(t, p=2, dim=-1)
def pack_one(t, pattern):
    return pack([t], pattern)
def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]
def cdist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x2 = reduce(x ** 2, 'b n d -> b n', 'sum')
    y2 = reduce(y ** 2, 'b n d -> b n', 'sum')
    xy = torch.einsum('b i d, b j d -> b i j', x, y) * -2
    return (rearrange(x2, 'b i -> b i 1') + rearrange(y2, 'b j -> b 1 j') + xy).sqrt()
def batched_sample_vectors(samples: torch.Tensor, num: int) -> torch.Tensor:
    return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim=0)], dim=0)
def batched_bincount(x: torch.Tensor, *, minlength: int) -> torch.Tensor:
    batch, dtype, device = x.shape[0], x.dtype, x.device
    target = torch.zeros(batch, minlength, dtype=dtype, device=device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target
def batched_embedding(indices: torch.Tensor, embeds: torch.Tensor) -> torch.Tensor:
    batch, dim = indices.shape[1], embeds.shape[-1]
    indices = repeat(indices, 'h b n -> h b n d', d=dim)
    embeds = repeat(embeds, 'h c d -> h b c d', b=batch)
    return embeds.gather(2, indices)
def kmeans(samples: torch.Tensor, num_clusters: int, num_iters: int = 100, use_cosine_sim: bool = False, sample_fn: Callable = batched_sample_vectors, all_reduce_fn: Callable = noop) -> (torch.Tensor, torch.Tensor):
    num_codebooks = samples.shape[0]
    dim = samples.shape[-1]
    dtype = samples.dtype
    means = sample_fn(samples, num_clusters)
    bins = None
    for _ in range(num_iters):
        if use_cosine_sim:
            dists = torch.einsum('h i d, h j d -> h i j', samples, means)
        else:
            dists = -cdist(samples, means)
        buckets = torch.argmax(dists, dim=-1)
        bins = batched_bincount(buckets, minlength=num_clusters)
        all_reduce_fn(bins)
        zero_mask = torch.Tensor(bins == 0)
        bins_min_clamped = bins.masked_fill(zero_mask, 1)
        new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(1, repeat(buckets, 'h n -> h n d', d=dim), samples)
        new_means = new_means / rearrange(bins_min_clamped, '... -> ... 1')
        all_reduce_fn(new_means)
        if use_cosine_sim:
            new_means = l2norm(new_means)
        means = torch.where(rearrange(zero_mask, "... -> ... 1"), means, new_means)
    return means, bins
def pad_shape(shape, size, dim=0):
    return [size if i == dim else s for i, s in enumerate(shape)]
def sample_multinomial(total_count: int, probs: torch.Tensor) -> torch.Tensor:
    device = probs.device
    probs = probs.cpu()
    total_count = probs.new_full((), total_count)
    remainder = probs.new_ones(())
    sample = torch.empty_like(probs, dtype=torch.long)
    for i, p in enumerate(probs):
        s = torch.binomial(total_count, p / remainder)
        sample[i] = s
        total_count -= s
        remainder -= p
    return sample.to(device)
def all_gather_sizes(x: torch.Tensor, dim: int) -> torch.Tensor:
    size = torch.tensor(x.shape[dim], dtype=torch.long, device=x.device)
    all_sizes = [torch.empty_like(size) for _ in range(dist.get_world_size())]
    dist.all_gather(all_sizes, size)
    return torch.stack(all_sizes)
def all_gather_variably_sized(x: torch.Tensor, sizes: Iterable[torch.Size], dim: int = 0) -> [torch.Tensor]:
    rank = dist.get_rank()
    all_x = []
    for i, size in enumerate(sizes):
        t = x if i == rank else x.new_empty(pad_shape(x.shape, size, dim))
        dist.broadcast(t, src=i, async_op=True)
        all_x.append(t)
    dist.barrier()
    return all_x
def sample_vectors_distributed(local_samples: torch.Tensor, num: int) -> torch.Tensor:
    local_samples = rearrange(local_samples, '1 ... -> ...')
    rank = dist.get_rank()
    all_num_samples = all_gather_sizes(local_samples, dim=0)
    if rank == 0:
        samples_per_rank = sample_multinomial(num, all_num_samples / all_num_samples.sum())
    else:
        samples_per_rank = torch.empty_like(all_num_samples)
    dist.broadcast(samples_per_rank, src=0)
    samples_per_rank = samples_per_rank.tolist()
    local_samples = sample_vectors(local_samples, samples_per_rank[rank])
    all_samples = all_gather_variably_sized(local_samples, samples_per_rank, dim=0)
    out = torch.cat(all_samples, dim=0)
    return rearrange(out, '... -> 1 ...')
def orthogonal_loss_fn(t):
    h, n = t.shape[:2]
    normed_codes = l2norm(t)
    cosine_sim = torch.einsum('h i d, h j d -> h i j', normed_codes, normed_codes)
    return (cosine_sim ** 2).sum() / (h * n ** 2) - (1 / n)
def round_up_multiple(num, mult):
    return math.ceil(num / mult) * mult