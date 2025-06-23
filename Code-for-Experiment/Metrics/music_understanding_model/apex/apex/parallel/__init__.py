import torch

if hasattr(torch.distributed, 'ReduceOp'):
    ReduceOp = torch.distributed.ReduceOp
elif hasattr(torch.distributed, 'reduce_op'):
    ReduceOp = torch.distributed.reduce_op
else:
    ReduceOp = torch.distributed.deprecated.reduce_op

from .distributed import DistributedDataParallel, Reducer

try:
    import syncbn
    from .optimized_sync_batchnorm import SyncBatchNorm
except ImportError as err:
    from Code_for_Experiment.Metrics.music_understanding_model.apex.apex.parallel.optimized_sync_batchnorm import SyncBatchNorm
    SyncBatchNorm.syncbn_import_error = err

def convert_syncbn_model(module, process_group=None, channel_last=False):
    mod = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        mod = SyncBatchNorm(module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats, process_group, channel_last=channel_last)
        mod.running_mean = module.running_mean
        mod.running_var = module.running_var
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()
    for name, child in module.named_children():
        mod.add_module(name, convert_syncbn_model(child, process_group=process_group, channel_last=channel_last))
    del module
    return mod

def create_syncbn_process_group(group_size):
    if group_size == 0:
        return None
    world_size = torch.distributed.get_world_size()
    assert world_size >= group_size
    assert world_size % group_size == 0
    group = None
    for group_num in range(world_size // group_size):
        group_ids = range(group_num * group_size, (group_num + 1) * group_size)
        cur_group = torch.distributed.new_group(ranks=group_ids)
        if torch.distributed.get_rank() // group_size == group_num:
            group = cur_group
    assert group is not None
    return group