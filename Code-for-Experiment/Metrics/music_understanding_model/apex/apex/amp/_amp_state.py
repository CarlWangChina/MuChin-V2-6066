import os
import torch

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if TORCH_MAJOR == 0:
    import collections.abc as container_abc
else:
    from torch._six import container_abc

class AmpState(object):
    def __init__(self):
        self.hard_override = False
        self.allow_incoming_model_not_fp32 = False
        self.verbosity = 1

_amp_state = AmpState()

def warn_or_err(msg):
    if _amp_state.hard_override:
        print("Warning:  " + msg)
    else:
        raise RuntimeError(msg)

distributed = False
if 'WORLD_SIZE' in os.environ:
    distributed = int(os.environ['WORLD_SIZE']) > 1

def maybe_print(msg, rank0=False):
    if _amp_state.verbosity > 0:
        if rank0:
            if distributed:
                if torch.distributed.get_rank() == 0:
                    print(msg)
            else:
                print(msg)
        else:
            print(msg)

def master_params(optimizer):
    for group in optimizer.param_groups:
        for p in group['params']:
            yield p