import torch
from torch._six import string_classes
import functools
import numpy as np
import warnings
from ._amp_state import _amp_state, warn_or_err, container_abcs
from .handle import disable_casts
from .scaler import LossScaler
from ._process_optimizer import _process_optimizer
from apex.fp16_utils import convert_network
from ..fp16_utils import FP16_Optimizer as FP16_Optimizer_general
from ..optimizers import FP16_Optimizer as FP16_Optimizer_for_fused
from ..optimizers import FusedAdam
from ..parallel import DistributedDataParallel as apex_DDP
from Code_for_Experiment.Metrics.music_understanding_model.apex.apex.parallel.LARC import LARC

def to_type(dtype, t):
    if isinstance(t, torch.Tensor):
        if not t.is_cuda:
            warnings.warn("An input tensor was not cuda.")
        if t.is_floating_point():
            return t.to(dtype)
        return t
    else:
        return t.to(dtype)

def applier(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    elif isinstance(value, string_classes):
        return value
    elif isinstance(value, np.ndarray):
        return value
    elif hasattr(value, "to"):
        return fn(value)
    elif isinstance(value, container_abcs.Mapping):
        return {applier(k, fn): applier(v, fn) for k, v in value.items()}
    elif isinstance(value, container_abcs.Iterable):
        return type(value)(applier(v, fn) for v in value)
    else:
        return value

def check_models(models):
    for model in models:
        parallel_type = None
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            parallel_type = "torch.nn.parallel.DistributedDataParallel"
        if isinstance(model, apex_DDP):
            parallel_type = "apex.parallel.DistributedDataParallel"
        if isinstance(model, torch.nn.parallel.DataParallel):
            parallel_type = "torch.nn.parallel.DataParallel"
        if parallel_type is not None:
            raise RuntimeError("Incoming model is an instance of {}. ".format(parallel_type) +
                               "Parallel wrappers should only be applied to the model(s) AFTER the model(s) have been returned from amp.initialize.")

def check_params_fp32(models):
    for model in models:
        for name, param in model.named_parameters():
            if param.is_floating_point():
                if 'Half' in param.type():
                    warn_or_err("Found param {} with type {}, expected torch.cuda.FloatTensor.\nWhen using amp.initialize, you do not need to call .half() on your model\nbefore passing it, no matter what optimization level you choose.".format(
                                name, param.type()))
                elif not param.is_cuda:
                    warn_or_err("Found param {} with type {}, expected torch.cuda.FloatTensor.\nWhen using amp.initialize, you need to provide a model with parameters\nlocated on a CUDA device before passing it no matter what optimization level\nyou chose. Use model.to('cuda') to use the default device.".format(
                                name, param.type()))
        if hasattr(model, 'named_buffers'):
            buf_iter = model.named_buffers()
        else:
            buf_iter = model._buffers
        for obj in buf_iter:
            if type(obj) == tuple:
                name, buf = obj
            else:
                name, buf = obj, buf_iter[obj]
            if buf.is_floating_point():
                if 'Half' in buf.type():
                    warn_or_err("Found buffer {} with type {}, expected torch.cuda.FloatTensor.\nWhen using amp.initialize, you do not need to call .half() on your model\nbefore passing it, no matter what optimization level you choose.".format(
                                name, buf.type()))
                elif not buf.is_cuda:
                    warn_or_err("Found buffer {} with type {}, expected torch.cuda.FloatTensor.\nWhen using amp.initialize, you need to provide a model with buffers\nlocated on a CUDA device before passing it no matter what optimization level\nyou chose. Use model.to('cuda') to use the default device.".format(
                                name, buf.type()))

def check_optimizers(optimizers):
    for optim in optimizers:
        bad_optim_type = None
        if isinstance(optim, FP16_Optimizer_general):
            bad_optim_type = "apex.fp16_utils.FP16_Optimizer"
        if isinstance(optim, FP16_Optimizer_for_fused):
            bad_optim_type = "apex.optimizers.FP16_Optimizer"
        if bad_optim_type is not None:
            raise RuntimeError("An incoming optimizer is an instance of {}. ".format(bad_optim_type) +
                               "The optimizer(s) passed to amp.initialize() must be bare instances of either ordinary Pytorch optimizers, or Apex fused optimizers (currently just FusedAdam, but FusedSGD will be added soon).  You should not manually wrap your optimizer in either apex.fp16_utils.FP16_Optimizer or apex.optimizers.FP16_Optimizer. amp.initialize will take care of that for you (if necessary) based on the specified opt_level (and optional overridden properties).")

def wrap_fused_adam(optimizer, properties):
    msg = 'Currently, the usage of FusedAdam is restricted to amp.initialize(..., opt_level="O2", keep_batchnorm_fp32=False, loss_scale=float or "dynamic").  We are working on enabling more general usage.'
    assert properties.master_weights is True, msg
    assert properties.cast_model_type is torch.float16, msg
    assert (properties.keep_batchnorm_fp32 is False or properties.keep_batchnorm_fp32 is None), msg
    if properties.loss_scale == "dynamic":
        return FP16_Optimizer_for_fused(optimizer, dynamic_loss_scale=True)
    else:
        return FP16_Optimizer_for_fused(optimizer, static_loss_scale=properties.loss_scale)

def _initialize(models, optimizers, properties, num_losses=1, cast_model_outputs=None):
    from apex.parallel import DistributedDataParallel as apex_DDP
    import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.parallel.__init__
    optimizers_was_list = False
    if isinstance(optimizers, torch.optim.Optimizer) or isinstance(optimizers, LARC):
        optimizers = [optimizers]
    elif optimizers is None:
        optimizers = []
    elif isinstance(optimizers, list):
        optimizers_was_list = True
        check_optimizers(optimizers)
    else:
        check_optimizers([optimizers])
        raise TypeError("optimizers must be either a single optimizer or a list of optimizers.")
    if isinstance(models, torch.nn.Module):
        models_was_list = False
        models = [models]
    elif isinstance(models, list):
        models_was_list = True
    else:
        raise TypeError("models must be either a single model or a list of models.")
    check_models(models)
    if not _amp_state.allow_incoming_model_not_fp32:
        check_params_fp32(models)
    if properties.cast_model_type:
        if properties.keep_batchnorm_fp32:
            for model in models:
                convert_network(model, properties.cast_model_type)
        else:
            for model in models:
                model.to(properties.cast_model_type)
        input_caster = functools.partial(to_type, properties.cast_model_type)
        if cast_model_outputs is not None:
            output_caster = functools.partial(to_type, cast_model_outputs)
        else:
            output_caster = functools.partial(to_type, torch.float32)
        for model in models:
            def patch_forward(old_fwd):
                def new_fwd(*args, **kwargs):
                    output = old_fwd(*applier(args, input_caster), **applier(kwargs, input_caster))
                    return applier(output, output_caster)
                return new_fwd
            model.forward = patch_forward(model.forward)
        for optimizer in optimizers:
            optimizer.load_state_dict(optimizer.state_dict())
    elif cast_model_outputs is not None:
        output_caster = functools.partial(to_type, cast_model_outputs)
        for model in models:
            def patch_forward(old_fwd):
                def new_fwd(*args, **kwargs):
                    output = old_fwd(*args, **kwargs)
                    return applier(output, output_caster)
                return new_fwd
            model.forward = patch_forward(model.forward)
    for i, optimizer in enumerate(optimizers):
        if isinstance(optimizer, FusedAdam):
            optimizers[i] = wrap_fused_adam(optimizer, properties)
        else:
            optimizers[i] = _process_optimizer(optimizer, properties)
    _amp_state.loss_scalers = []
    for _ in range(num_losses):
        _amp_state.loss_scalers.append(LossScaler(properties.loss_scale, min_loss_scale=_amp_state.min_loss_scale, max_loss_scale=_amp_state.max_loss_scale))
    if properties.patch_torch_functions:
        from apex.amp import amp_init
        handle = amp_init(loss_scale=properties.loss_scale, verbose=(_amp_state.verbosity == 2))
        for optimizer in optimizers:
            def patch_step(old_step):
                def new_step(*args, **kwargs):
                    with disable_casts():
                        output = old_step(*args, **kwargs)
                    return output
                return new_step
            optimizer.step = patch_step(optimizer.step)
    if optimizers_was_list:
        if models_was_list:
            return models, optimizers
        else:
            return models[0], optimizers
    else:
        if models_was_list:
            if len(optimizers) == 0:
                return models
            else:
                return models, optimizers[0]
        else:
            if len(optimizers) == 0:
                return models[0]
            else:
                return models[0], optimizers[0]