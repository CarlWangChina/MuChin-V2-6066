import contextlib
import warnings
import torch
from . import utils
from .opt import OptimWrapper
from .scaler import LossScaler
from ._amp_state import _amp_state, master_params, maybe_print
from ..fp16_utils import FP16_Optimizer as FP16_Optimizer_general
from ..optimizers import FP16_Optimizer as FP16_Optimizer_for_fused
from Code_for_Experiment.Metrics.music_understanding_model.apex.apex.parallel.LARC import LARC

@contextlib.contextmanager
def scale_loss(loss, optimizers, loss_id=0, model=None, delay_unscale=False, delay_overflow_check=False):
    if not hasattr(_amp_state, "opt_properties"):
        raise RuntimeError("Invoked 'with amp.scale_loss`, but internal Amp state has not been initialized.  "
                           "model, optimizer = amp.initialize(model, optimizer, opt_level=...) must be called "
                           "before `with amp.scale_loss`.")
    if not _amp_state.opt_properties.enabled:
        yield loss
        return
    if isinstance(optimizers, torch.optim.Optimizer) or isinstance(optimizers, LARC):
        optimizers = [optimizers]
    if isinstance(optimizers, FP16_Optimizer_for_fused):
        loss_scale = optimizers.cur_scale
    else:
        loss_scaler = _amp_state.loss_scalers[loss_id]
        loss_scale = loss_scaler.loss_scale()
    if ((not _amp_state.opt_properties.master_weights) and
            (not loss_scaler.dynamic) and
            loss_scale == 1.0):
        yield loss.float()
        if _amp_state.opt_properties.patch_torch_functions:
            _amp_state.handle._clear_cache()
        return
    if not delay_unscale:
        if isinstance(optimizers, list):
            for optimizer in optimizers:
                if not optimizer._amp_stash.params_have_scaled_gradients:
                    optimizer._prepare_amp_backward()
    yield (loss.float()) * loss_scale
    if delay_unscale:
        for optimizer in optimizers:
            optimizer._amp_stash.params_have_scaled_gradients = True
    else:
        if not isinstance(optimizers, FP16_Optimizer_for_fused):
            loss_scaler.clear_overflow_state()
            for optimizer in optimizers:
                optimizer._post_amp_backward(loss_scaler)
                optimizer._amp_stash.params_have_scaled_gradients = False
            should_skip = False if delay_overflow_check else loss_scaler.update_scale()
            if should_skip:
                for optimizer in optimizers:
                    if not optimizer._amp_stash.already_patched:
                        def patch_step(opt, loss_scaler, loss_id):
                            opt_step = opt.step
                            def skip_step(closure=None):
                                if closure is not None:
                                    raise RuntimeError("Currently, Amp does not support closure use with optimizers.")
                                maybe_print(("Gradient overflow.  Skipping step, loss scaler " +
                                             "{} reducing loss scale to {}").format(loss_id,
                                                                                    loss_scaler.loss_scale()))
                                if hasattr(opt._amp_stash, "all_fp32_from_fp16_params"):
                                    for param in opt._amp_stash.all_fp32_from_fp16_params:
                                        param.grad = None
                                opt.step = opt_step
                                opt._amp_stash.already_patched = False
                            return skip_step
                        optimizer.step = patch_step(optimizer, loss_scaler, loss_id)
                        optimizer._amp_stash.already_patched = True
    if _amp_state.opt_properties.patch_torch_functions:
        _amp_state.handle._clear_cache()

@contextlib.contextmanager
def disable_casts():
    _amp_state.handle._is_active = False
    yield
    _amp_state.handle._is_active = True

class AmpHandle(object):
    def __init__(self, loss_scale="dynamic", enable_caching=True, verbose=False):
        self._enable_caching = enable_caching
        self._verbose = verbose
        self._cache = dict()
        self._default_scaler = LossScaler(loss_scale)
        self._is_active = True
        self._all_wrappers = []

    def is_active(self):
        return self._is_active

    @contextlib.contextmanager
    def _disable_casts(self):
        self._is_active = False
        yield
        self._is_active = True

    def wrap_optimizer(self, optimizer, num_loss=1):
        self._default_scaler = None
        return OptimWrapper(optimizer, self, num_loss)

    @contextlib.contextmanager
    def scale_loss(self, loss, optimizer):
        raise RuntimeError("The old Amp API is no longer supported.  Please move to the new API, "
                           "documented here:  https://nvidia.github.io/apex/amp.html.  Transition guide:  "
                           "https://nvidia.github.io/apex/amp.html")
        if not self.is_active():
            yield loss
            return
        if self._default_scaler is None:
            raise RuntimeError(
                'After calling `handle.wrap_optimizer()`, you must explicitly ' +
                'use `optimizer.scale_loss(loss)`.')
        loss_scale = self._default_scaler.loss_scale()
        yield loss * loss_scale
        self._default_scaler.clear_overflow_state()
        self._default_scaler.unscale(
            master_params(optimizer),
            master_params(optimizer),
            loss_scale)
        should_skip = self._default_scaler.update_scale()
        if should_skip:
            optimizer_step = optimizer.step
            def skip_step():
                maybe_print('Gradient overflow, skipping update')
                optimizer.step = optimizer_step
            optimizer.step = skip_step
        self._clear_cache()

    def _clear_cache(self):
        self._cache.clear()

    def _save_func(self, mod, fn, func):
        self._all_wrappers.append((mod, fn, func))

    def _deactivate(self):
        for mod, fn, func in self._all_wrappers:
            utils.set_func(mod, fn, func)
        self._all_wrappers = []

    @property
    def has_cache(self):
        return self._enable_caching

    @property
    def cache(self):
        return self._cache

    def remove_cache(self, param):
        if self.has_cache and param in self.cache:
            del self.cache[param]

    @property
    def verbose(self):
        return self._verbose

class NoOpHandle(object):
    def is_active(self):
        return False

    @contextlib.contextmanager
    def _disable_casts(self):
        yield

    def wrap_optimizer(self, optimizer, num_loss=1):
        return OptimWrapper(optimizer, self, num_loss)

    @contextlib.contextmanager
    def scale_loss(self, loss, optimizer):
        yield loss

    @property
    def has_cache(self):
        return False

    @property
    def verbose(self):
        return False

    def _clear_cache(self):
        pass

    def _deactivate(self):
        pass