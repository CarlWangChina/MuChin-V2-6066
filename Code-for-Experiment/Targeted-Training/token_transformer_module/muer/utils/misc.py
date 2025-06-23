import torch
from typing import Optional

def merge_tensors(tensors: [torch.Tensor], *, padding: int or float = 0, dtype: Optional[torch.dtype] = None, device: Optional[str or torch.device] = None) -> torch.Tensor:
    if len(tensors) == 0:
        return torch.Tensor()
    if dtype is None:
        dtype = tensors[0].dtype
    if device is None:
        device = tensors[0].device
    shape = tensors[0].shape
    assert len(shape) >= 2, "The dimension of the input tensor should be at least 2."
    assert all([t.shape[2:] == shape[2:] for t in tensors]), "The input tensors should have the same shape."
    count = sum([t.shape[0] for t in tensors])
    merged_shape = [count, max([t.shape[1] for t in tensors])]
    merged_shape.extend(shape[2:])
    merged_tensor = torch.full(merged_shape, fill_value=padding, dtype=dtype, device=device)
    i = 0
    for batch in range(len(tensors)):
        merged_tensor[i:i + tensors[batch].shape[0], :tensors[batch].shape[1], ...] = tensors[batch]
        i += tensors[batch].shape[0]
    return merged_tensor

def init_embedding_(embedding_layer):
    with torch.no_grad():
        weight = embedding_layer.weight
        norm = weight.norm(p=2, dim=1, keepdim=True)
        weight.div_(norm)

def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    results = []
    for batch in range(probs_sort.shape[0]):
        next_token = torch.multinomial(probs_sort[batch], num_samples=1)
        next_token = torch.gather(probs_idx[batch], -1, next_token)
        results.append(next_token)
    return torch.stack(results, dim=0).squeeze(-1)

def safe_softmax(x: torch.Tensor, dim: Optional[int] = None, dtype: Optional[torch.dtype] = None, eps: float = 1e-7) -> torch.Tensor:
    if dtype is not None:
        original_dtype = x.dtype
        x = x.to(dtype)
    else:
        original_dtype = None
    if dim is None:
        x = x - x.max()
        x.exp_()
        x = x / (x.sum() + eps)
    else:
        x = x - x.max(dim=dim, keepdim=True).values
        x.exp_()
        x = x / (x.sum(dim=dim, keepdim=True) + eps)
    if dtype is not None:
        x = x.to(original_dtype)
    return x

def safe_softmax_1(x: torch.Tensor, dim: Optional[int] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    return safe_softmax(x, dim=dim, dtype=dtype, eps=1.0)