import torch
from enum import Enum
from typing import Iterable, List
from contextlib import contextmanager
from functools import partial, wraps
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.package import is_package_installed

def is_deepspeed_installed() -> bool:
    return is_package_installed("deepspeed")

class ReduceOp(Enum):
    SUM = 0
    PRODUCT = 1
    MIN = 2
    MAX = 3
    AVG = 4

    def to_dist_op(self):
        if is_deepspeed_installed():
            import deepspeed.comm as dist
        else:
            import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.parallel.distributed as dist
        return {
            self.SUM: dist.ReduceOp.SUM,
            self.PRODUCT: dist.ReduceOp.PRODUCT,
            self.MIN: dist.ReduceOp.MIN,
            self.MAX: dist.ReduceOp.MAX,
            self.AVG: dist.ReduceOp.AVG
        }[self]

def is_available():
    if is_deepspeed_installed():
        import deepspeed.comm as dist
    else:
        import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.parallel.distributed as dist
    return dist.is_available()

def is_initialized():
    if is_deepspeed_installed():
        import deepspeed.comm as dist
    else:
        import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.parallel.distributed as dist
    return dist.is_available() and dist.is_initialized()

def get_rank():
    if is_initialized():
        if is_deepspeed_installed():
            import deepspeed.comm as dist
        else:
            import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.parallel.distributed as dist
        return dist.get_rank()
    else:
        if torch.cuda.is_available():
            return torch.cuda.current_device()
        else:
            return 0

def is_primary():
    return get_rank() == 0

def get_local_rank():
    if is_initialized():
        if is_deepspeed_installed():
            import deepspeed.comm as dist
            return dist.get_local_rank()
        else:
            import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.parallel.distributed as dist
            return dist.get_rank()
    else:
        if torch.cuda.is_available():
            return torch.cuda.current_device()
        else:
            return 0

def get_world_size():
    if is_initialized():
        if is_deepspeed_installed():
            import deepspeed.comm as dist
        else:
            import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.parallel.distributed as dist
        return dist.get_world_size()
    else:
        return 1

def barrier():
    if is_initialized():
        if is_deepspeed_installed():
            import deepspeed.comm as dist
        else:
            import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.parallel.distributed as dist
        dist.barrier()

def broadcast(tensor, src=0, async_op=False):
    if is_initialized():
        if is_deepspeed_installed():
            import deepspeed.comm as dist
        else:
            import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.parallel.distributed as dist
        return dist.broadcast(tensor, src=src, async_op=async_op)

def broadcast_tensors(tensors: Iterable[torch.Tensor], src: int = 0):
    if not is_initialized():
        return
    tensors = [tensor for tensor in tensors if _is_complex_or_float(tensor)]
    _check_number_of_params(tensors)
    handles = []
    for tensor in tensors:
        handle = broadcast(tensor.data, src=src, async_op=True)
        handles.append(handle)
    for handle in handles:
        handle.wait()

def broadcast_model(model: torch.nn.Module, src: int = 0):
    broadcast_tensors(model.parameters(), src=src)
    broadcast_tensors(model.buffers(), src=src)

def reduce(tensor, dst=0, op=ReduceOp.SUM):
    if is_initialized():
        if is_deepspeed_installed():
            import deepspeed.comm as dist
        else:
            import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.parallel.distributed as dist
        return dist.reduce(tensor, dst=dst, op=op.to_dist_op())

def all_reduce(tensor, op=ReduceOp.SUM, async_op=False):
    if is_initialized():
        if is_deepspeed_installed():
            import deepspeed.comm as dist
        else:
            import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.parallel.distributed as dist
        return dist.all_reduce(tensor, op=op.to_dist_op(), async_op=async_op)

def _is_complex_or_float(tensor):
    return torch.is_floating_point(tensor) or torch.is_complex(tensor)

def _check_number_of_params(params: List[torch.Tensor]):
    if not is_initialized() or not params:
        return
    tensor = torch.tensor([len(params)], device=params[0].device, dtype=torch.long)
    all_reduce(tensor)
    if tensor.item() != len(params) * get_world_size():
        raise RuntimeError(f"Mismatch in number of params: ours is {len(params)}, "
                           "at least one worker has a different one.")

def average_tensors(tensors: Iterable[torch.Tensor]):
    if not is_initialized():
        return tensors
    tensors = [tensor for tensor in tensors if _is_complex_or_float(tensor)]
    _check_number_of_params(tensors)
    handles = []
    for tensor in tensors:
        handle = all_reduce(
            tensor.data, op=ReduceOp.SUM, async_op=True)
        handles.append((tensor, handle))
    for tensor, handle in handles:
        handle.wait()
        tensor.data /= get_world_size()

def sync_gradients(params: Iterable[torch.Tensor]):
    grads = [param.grad for param in params if param.grad is not None]
    average_tensors(grads)

@contextmanager
def eager_sync_gradients(params: Iterable[torch.Tensor]):
    if not is_initialized():
        yield
        return
    params = list([p for p in params if p.requires_grad])
    _check_number_of_params(params)
    hooks = []
    handles = []
    waiting_params = set(params)
    def _callback(param, grad):
        if param not in waiting_params:
            raise RuntimeError(f"We got a gradient twice for parameter {param}.")
        _handle = all_reduce(grad.data, op=ReduceOp.SUM, async_op=True)
        handles.append((param, grad.data, _handle))
        waiting_params.remove(param)
    for param in params:
        hooks.append(param.register_hook(partial(_callback, param)))
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()
        _check_number_of_params(list(waiting_params))
        for param, grad, handle in handles:
            handle.wait()
            assert param.grad is not None
            torch.div(grad, get_world_size(), out=param.grad)

def sync_model(model: torch.nn.Module, sync_buffers: bool = True, average_buffers: bool = True):
    sync_gradients(model.parameters())
    if sync_buffers:
        if average_buffers:
            average_tensors(model.buffers())
        else:
            broadcast_tensors(model.buffers())

@contextmanager
def eager_sync_model(model: torch.nn.Module, sync_buffers: bool = True, average_buffers: bool = True):
    with eager_sync_gradients(model.parameters()):
        yield
    if sync_buffers:
        if average_buffers:
            average_tensors(model.buffers())
        else:
            broadcast_tensors(model.buffers())

def get_device() -> torch.device:
    if is_deepspeed_installed():
        from deepspeed.accelerator import get_accelerator
        if get_accelerator().is_available():
            return torch.device(get_accelerator().device_name(), get_local_rank())
    if torch.cuda.is_available():
        return torch.device(f"cuda:{get_local_rank()}")
    elif torch.backends.mps.is_available():
        return torch.device("mps:0")
    else:
        return torch.device("cpu")