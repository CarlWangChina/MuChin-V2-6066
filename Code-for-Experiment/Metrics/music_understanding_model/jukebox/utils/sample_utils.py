import Code_for_Experiment.Metrics.music_understanding_model.tensorboardX.tests.test_pytorch_np
import torch as t

def split_batch(obj, n_samples, split_size):
    n_passes = (n_samples + split_size - 1) // split_size
    if isinstance(obj, t.Tensor):
        return t.split(obj, split_size, dim=0)
    elif isinstance(obj, list):
        return list(zip(*[t.split(item, split_size, dim=0) for item in obj]))
    elif obj is None:
        return [None] * n_passes
    else:
        raise TypeError('Unknown input type')

def get_starts(total_length, n_ctx, hop_length):
    starts = []
    for start in range(0, total_length - n_ctx + hop_length, hop_length):
        if start + n_ctx >= total_length:
            start = total_length - n_ctx
        starts.append(start)
    return starts