from contextlib import contextmanager
import functools
import hashlib
import inspect
import io
from pathlib import Path
import warnings
from omegaconf import OmegaConf
from Code_for_Experiment.Targeted_Training.audio_quality_screening.encodec.__main__ import fatal
import torch

def _check_diffq():
    try:
        import diffq
    except ImportError:
        fatal('Trying to use DiffQ, but diffq is not installed.\n'
              'On Windows run: python.exe -m pip install diffq \n'
              'On Linux/Mac, run: python3 -m pip install diffq')

def get_quantizer(model, args, optimizer=None):
    quantizer = None
    if args.diffq:
        _check_diffq()
        from diffq import DiffQuantizer
        quantizer = DiffQuantizer(
            model, min_size=args.min_size, group_size=args.group_size)
        if optimizer is not None:
            quantizer.setup_optimizer(optimizer)
    elif args.qat:
        _check_diffq()
        from diffq import UniformQuantizer
        quantizer = UniformQuantizer(
                model, bits=args.qat, min_size=args.min_size)
    return quantizer

def load_model(path_or_package, strict=False):
    if isinstance(path_or_package, dict):
        package = path_or_package
    elif isinstance(path_or_package, (str, Path)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path = path_or_package
            package = torch.load(path, map_location='cpu')
    else:
        raise ValueError(f"Invalid type for {path_or_package}.")
    klass = package["klass"]
    args = package["args"]
    kwargs = package["kwargs"]
    if strict:
        model = klass(*args, **kwargs)
    else:
        sig = inspect.signature(klass)
        for key in list(kwargs):
            if key not in sig.parameters:
                warnings.warn("Dropping inexistant parameter " + key)
                del kwargs[key]
        model = klass(*args, **kwargs)
    state = package["state"]
    set_state(model, state)
    return model

def get_state(model, quantizer, half=False):
    if quantizer is None:
        dtype = torch.half if half else None
        state = {k: p.data.to(device='cpu', dtype=dtype) for k, p in model.state_dict().items()}
    else:
        state = quantizer.get_quantized_state()
        state['__quantized'] = True
    return state

def set_state(model, state, quantizer=None):
    if state.get('__quantized'):
        if quantizer is not None:
            quantizer.restore_quantized_state(model, state['quantized'])
        else:
            _check_diffq()
            from diffq import restore_quantized_state
            restore_quantized_state(model, state)
    else:
        model.load_state_dict(state)
    return state

def save_with_checksum(content, path):
    buf = io.BytesIO()
    torch.save(content, buf)
    sig = hashlib.sha256(buf.getvalue()).hexdigest()[:8]
    path = path.parent / (path.stem + "-" + sig + path.suffix)
    path.write_bytes(buf.getvalue())

def serialize_model(model, training_args, quantizer=None, half=True):
    args, kwargs = model._init_args_kwargs
    klass = model.__class__
    state = get_state(model, quantizer, half)
    return {
        'klass': klass,
        'args': args,
        'kwargs': kwargs,
        'state': state,
        'training_args': OmegaConf.to_container(training_args, resolve=True),
    }

def copy_state(state):
    return {k: v.cpu().clone() for k, v in state.items()}

@contextmanager
def swap_state(model, state):
    old_state = copy_state(model.state_dict())
    model.load_state_dict(state, strict=False)
    try:
        yield
    finally:
        model.load_state_dict(old_state)

def capture_init(init):
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)
    return __init__