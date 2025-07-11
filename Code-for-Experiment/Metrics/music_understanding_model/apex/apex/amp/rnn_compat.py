from . import utils, wrap

import torch

_VF = torch._C._VariableFunctions

RNN_NAMES = ['rnn_relu', 'rnn_tanh', 'gru', 'lstm']

def _gen_VF_wrapper(name):
    def wrapper(*args, **kwargs):
        return getattr(_VF, name)(*args, **kwargs)
    return wrapper

class VariableFunctionsShim(object):
    def __init__(self):
        for name in RNN_NAMES:
            for suffix in ['', '_cell']:
                fn_name = name + suffix
                setattr(self, fn_name, _gen_VF_wrapper(fn_name))

def has_old_rnns():
    try:
        torch.nn.backends.thnn.backend.LSTMCell
        return True
    except AttributeError:
        return False

def whitelist_rnn_cells(handle, verbose):
    if has_old_rnns():
        fn_names = ['RNNReLUCell', 'RNNTanhCell', 'LSTMCell', 'GRUCell']
        mod = torch.nn.backends.thnn.backend
    else:
        fn_names = [x + '_cell' for x in RNN_NAMES]
        mod = torch.nn.modules.rnn._VF
        assert isinstance(mod, VariableFunctionsShim)
    for fn in fn_names:
        wrap.cached_cast(mod, fn, utils.maybe_half, handle, try_caching=True, verbose=verbose)
    if has_old_rnns():
        for rnn_type in ['GRUFused', 'LSTMFused']:
            mod = getattr(torch.nn._functions.thnn.rnnFusedPointwise, rnn_type)
            wrap.disable_casts(mod, 'backward', handle)