import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.amp.rnn_compat
import torch.nn as nn
import torch.nn.functional as F

def normalization(channels: int) -> nn.Module:
    return nn.GroupNorm(num_channels=channels, num_groups=32, eps=1e-4)

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

def avg_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")