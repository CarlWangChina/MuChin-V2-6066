import logging
import pickle
import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset
from Code_for_Experiment.Metrics.music_understanding_model.apex.apex.parallel.distributed import DistributedDataParallel

logger = logging.getLogger(__name__)
rank = 0
world_size = 1

def init():
    global rank, world_size
    if not torch.distributed.is_initialized():
        dora_distrib.init()
    rank = dora_distrib.rank()
    world_size = dora_distrib.world_size()

def average(metrics, count=1.):
    if isinstance(metrics, dict):
        keys, values = zip(*sorted(metrics.items()))
        values = average(values, count)
        return dict(zip(keys, values))
    if world_size == 1:
        return metrics
    tensor = torch.tensor(list(metrics) + [1], device='cuda', dtype=torch.float32)
    tensor *= count
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return (tensor[:-1] / tensor[-1]).cpu().numpy().tolist()

def wrap(model):
    if world_size == 1:
        return model
    else:
        return DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device()
        )

def barrier():
    if world_size > 1:
        torch.distributed.barrier()

def share(obj=None, src=0):
    if world_size == 1:
        return obj
    size = torch.empty(1, device='cuda', dtype=torch.long)
    if rank == src:
        dump = pickle.dumps(obj)
        size[0] = len(dump)
    torch.distributed.broadcast(size, src=src)
    if rank == src:
        buffer = torch.from_numpy(np.frombuffer(dump, dtype=np.uint8).copy()).cuda()
    else:
        buffer = torch.empty(size[0].item(), device='cuda', dtype=torch.uint8)
    torch.distributed.broadcast(buffer, src=src)
    if rank != src:
        obj = pickle.loads(buffer.cpu().numpy().tobytes())
    logger.debug(f"Shared object of size {len(buffer)}")
    return obj

def loader(dataset, *args, shuffle=False, klass=DataLoader, **kwargs):
    if world_size == 1:
        return klass(dataset, *args, shuffle=shuffle, **kwargs)
    if shuffle:
        sampler = DistributedSampler(dataset)
        return klass(dataset, *args, **kwargs, sampler=sampler)
    else:
        dataset = Subset(dataset, list(range(rank, len(dataset), world_size)))
        return klass(dataset, *args, shuffle=shuffle, **kwargs)