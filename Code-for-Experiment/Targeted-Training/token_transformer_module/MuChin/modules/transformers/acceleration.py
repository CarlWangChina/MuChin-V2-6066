from torch import torch
from ama_prof_divi.utils.logging import get_logger
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.modules.transformers.model_args import TransformerModelArgs
Argslogger = get_logger(__name__)
class InferAccelerationCache:
    def __init__(self, args: TransformerModelArgs, device: str = 'cpu'):
        self.device = device
        self.num_layers = args.num_layers
        self.max_seq_len = args.max_seq_len
        if self.device != 'cpu':
            Argslogger.warning("Using GPU memory to store the cache will cause significant GPU memory overhead.")
        self.kv_cache_cross = None
        self.kv_cache_self = None
        self.kv_cache_cross_set = [False] * self.num_layers

    def cleanup(self):
        self.kv_cache_cross = None
        self.kv_cache_self = None
        self.kv_cache_cross_set = [False] * self.num_layers

    def update_kv_cache_cross(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        assert layer < self.num_layers, f"Layer index {layer} is out of range."
        assert k.dim() == 4, f"Input k must be 4-dimensional, got {k.dim()}."
        assert v.dim() == 4, f"Input v must be 4-dimensional, got {v.dim()}."
        assert k.shape == v.shape, f"Input k and v must have the same shape, got {k.shape} and {v.shape}."
        if self.kv_cache_cross is None:
            self.kv_cache_cross = torch.zeros((self.num_layers, 2, *k.shape), device=self.device)
            self.kv_cache_cross.requires_grad = False
            self.kv_cache_cross[layer][0] = k.to(self.device)
            self.kv_cache_cross[layer][1] = v.to(self.device)
        else:
            assert self.kv_cache_cross[layer][0].shape == k.shape, \
                (f"Input k must have the same shape as the cached k, got {k.shape} and {self.kv_cache_cross[layer][0].shape}.")
            assert self.kv_cache_cross[layer][1].shape == v.shape, \
                (f"Input k must have the same shape as the cached k, got {v.shape} and {self.kv_cache_cross[layer][1].shape}.")
            assert not self.kv_cache_cross_set[layer], \
                f"Cross attention cache for layer {layer} is already set."
            self.kv_cache_cross[layer][0] = k.to(self.device)
            self.kv_cache_cross[layer][1] = v.to(self.device)
        self.kv_cache_cross_set[layer] = True

    def get_kv_cache_cross(self, layer: int, device: str) -> (torch.Tensor, torch.Tensor):
        assert layer < self.num_layers, f"Layer index {layer} is out of range."
        assert self.kv_cache_cross is not None, "Cache is empty."
        assert self.kv_cache_cross_set[layer], f"Cross attention cache for layer {layer} is not set."
        return self.kv_cache_cross[layer][0].to(device), self.kv_cache_cross[layer][1].to(device)

    def is_kv_cache_cross_set(self, layer: int) -> bool:
        assert 0 <= layer < self.num_layers, f"Layer index {layer} is out of range."
        return self.kv_cache_cross_set[layer]

    def update_kv_cache_self(self, layer: int, k: torch.Tensor, v: torch.Tensor, start_pos: int):
        assert layer < self.num_layers, f"Layer index {layer} is out of range."
        assert k.dim() == 4, f"Input k must be 4-dimensional, got {k.dim()}."
        assert v.dim() == 4, f"Input v must be 4-dimensional, got {v.dim()}."
        assert k.shape == v.shape, f"Input k and v must have the same shape, got {k.shape} and {v.shape}."
        assert start_pos >= 0, f"Start position must be non-negative, got {start_pos}."
        assert start_pos + k.shape[1] <= self.max_seq_len, \
            f"End position k {start_pos + k.shape[1]} exceeds maximum sequence length {self.max_seq_len}."
        assert start_pos + v.shape[1] <= self.max_seq_len, \
            f"End position v {start_pos + v.shape[1]} exceeds maximum sequence length {self.max_seq_len}."
        if self.kv_cache_self is None:
            self.kv_cache_self = torch.zeros((self.num_layers, 2, k.shape[0], self.max_seq_len, k.shape[2], k.shape[3]), device=self.device)
            self.kv_cache_self.requires_grad = False
            self.kv_cache_self[layer][0][:, start_pos:start_pos + k.shape[1], :, :] = k.to(self.device)
            self.kv_cache_self[layer][1][:, start_pos:start_pos + v.shape[1], :, :] = v.to(self.device)
        else:
            assert self.kv_cache_self[layer][0].shape[0] == k.shape[0]
            assert self.kv_cache_self[layer][0].shape[2] == k.shape[2]
            assert self.kv_cache_self[layer][0].shape[3] == k.shape[3]
            self.kv_cache_self[layer][0][:, start_pos:start_pos + k.shape[1], :, :] = k.to(self.device)
            self.kv_cache_self[layer][1][:, start_pos:start_pos + v.shape[1], :, :] = v.to(self.device)

    def get_kv_cache_self(self, layer: int, start_pos: int, end_pos: int, device: str) -> (torch.Tensor, torch.Tensor):
        assert layer < self.num_layers, f"Layer index {layer} is out of range."
        assert self.kv_cache_self is not None, "Cache is empty."
        assert start_pos >= 0, f"Start position must be non-negative, got {start_pos}."
        assert end_pos <= self.max_seq_len, \
            f"End position {end_pos} exceeds maximum sequence length {self.max_seq_len}."
        return self.kv_cache_self[layer][0][:, start_pos:end_pos, :, :].to(device), \
            self.kv_cache_self[layer][1][:, start_pos:end_pos, :, :].to(device)