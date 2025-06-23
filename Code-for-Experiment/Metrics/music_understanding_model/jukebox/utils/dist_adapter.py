import torch.distributed as dist
from enum import Enum

class ReduceOp(Enum):
    SUM = 0
    PRODUCT = 1
    MIN = 2
    MAX = 3

    def ToDistOp(self):
        return {
            self.SUM: dist.ReduceOp.SUM,
            self.PRODUCT: dist.ReduceOp.PRODUCT,
            self.MIN: dist.ReduceOp.MIN,
            self.MAX: dist.ReduceOp.MAX
        }[self]

def is_available():
    return dist.is_available()

def get_rank():
    if is_available():
        return dist.get_rank()
    else:
        return 0

def get_world_size():
    if is_available():
        return dist.get_world_size()
    else:
        return 1

def barrier():
    if is_available():
        return dist.barrier()

def all_gather(tensor_list, tensor):
    if is_available():
        return dist.all_gather(tensor_list, tensor)
    else:
        tensor_list[0] = tensor

def all_reduce(tensor, op=ReduceOp.SUM):
    if is_available():
        return dist.all_reduce(tensor, op.ToDistOp())

def reduce(tensor, dst, op=ReduceOp.SUM):
    if is_available():
        return dist.reduce(tensor, dst, op.ToDistOp())

def broadcast(tensor, src):
    if is_available():
        return dist.broadcast(tensor, src)

def init_process_group(backend, init_method):
    if is_available():
        return dist.init_process_group(backend, init_method)