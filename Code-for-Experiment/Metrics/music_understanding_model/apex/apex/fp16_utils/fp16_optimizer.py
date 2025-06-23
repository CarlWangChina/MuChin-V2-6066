import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from ..amp._amp_state import _amp_state, maybe_print
from Code_for_Experiment.Metrics.music_understanding_model.apex.apex.fp16_utils.loss_scaler import LossScaler
from ..multi_tensor_apply import multi_tensor_applier
from .fp16util import model_grads_to_master_grads, master_params_to_model_params, clip_grad_norm

class FP16_Optimizer(object):
    def __init__(self,
                 init_optimizer,
                 static_loss_scale=1.0,
                 dynamic_loss_scale=False,
                 dynamic_loss_args=None,
                 verbose=True):
        if not torch.cuda.is_available():
            raise SystemError("Cannot use fp16 without CUDA.")
        self.verbose = verbose
        self.optimizer = init_optimizer
        self.fp16_groups = []
        self.fp32_from_fp16_groups = []
        self.fp32_from_fp32_groups = []
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.maybe_print("FP16_Optimizer processing param group {}:".format(i))
            fp16_params_this_group = []
            fp32_params_this_group = []
            fp32_from_fp16_params_this_group = []
            for i, param in enumerate(param_group['params']):
                if param.requires_grad:
                    if param.type() == 'torch.cuda.HalfTensor':
                        self.maybe_print("FP16_Optimizer received torch.cuda.HalfTensor with {}".format(param.size()))
                        fp16_params_this_group.append(param)
                        master_param = param.detach().clone().float()
                        master_param.requires_grad = True
                        param_group['params'][i] = master_param
                        fp32_from_fp16_params_this_group.append(master_param)
                        if param in self.optimizer.state:
                            self.optimizer.state[master_param] = self.optimizer.state.pop(param)
                    elif param.type() == 'torch.cuda.FloatTensor':
                        self.maybe_print("FP16_Optimizer received torch.cuda.FloatTensor with {}".format(param.size()))
                        fp32_params_this_group.append(param)
                        param_group['params'][i] = param
                    else:
                        raise TypeError("Wrapped parameters must be either torch.cuda.FloatTensor or torch.cuda.HalfTensor. Received {}".format(param.type()))
            self.fp16_groups.append(fp16_params_this_group)
            self.fp32_from_fp16_groups.append(fp32_from_fp16_params_this_group)
            self.fp32_from_fp32_groups.append(fp32_params_this_group)
        self.all_fp16_params = []
        for group in self.fp16_groups:
            self.all_fp16_params += group
        self.all_fp32_from_fp16_params = []
        for group in self.fp32_from_fp16_groups:
            self.all_fp32_from_fp16_params += group
        self.all_fp32_from_fp32_params = []
        for group in self.fp32_from_fp32_groups:
            self.all_fp32_from_fp32_params += group
        self.optimizer.load_state_dict(self.optimizer.state_dict())
        if dynamic_loss_scale:
            self.dynamic_loss_scale = True
            if dynamic_loss_args is not None:
                self.loss_scaler = LossScaler("dynamic", **dynamic_loss_args)
            else:
                self.loss_scaler = LossScaler("dynamic")
        else:
            self.dynamic_loss_scale = False
            self.loss_scaler = LossScaler(static_loss_scale)
        self.overflow = False
        self.first_closure_call_this_step = True
        self.clip_grad_norm = clip_grad_norm
        if multi_tensor_applier.available:
            import amp_C
            self.multi_tensor_scale = amp_C.multi_tensor_scale
            self._dummy_overflow_buf = torch.cuda.IntTensor([0])

    def maybe_print(self, msg):
        if self.verbose:
            print(msg)

    def __getstate__(self):
        raise RuntimeError("FP16_Optimizer should be serialized using state_dict().")

    def __setstate__(self, state):
        raise RuntimeError("FP16_Optimizer should be deserialized using load_state_dict().")

    def zero_grad(self, set_grads_to_None=False):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if set_grads_to_None:
                    p.grad = None
                else:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()
        for fp16_group in self.fp16_groups:
            for param in fp16_group:
                if set_grads_to_None:
                    param.grad = None
                else:
                    if param.grad is not None:
                        param.grad.detach_()
                        param.grad.zero_()

    def _master_params_to_model_params(self):
        if multi_tensor_applier.available:
            if len(self.all_fp16_params) > 0:
                multi_tensor_applier(
                    self.multi_tensor_scale,
                    self._dummy_overflow_buf,
                    [self.all_fp32_from_fp16_params, self.all_fp16_params],
                    1.0)
        else:
            for fp16_group, fp32_from_fp16_group in zip(self.fp16_groups, self.fp32_from_fp16_groups):
                master_params_to_model_params(fp16_group, fp32_from_fp16_group)

    def clip_master_grads(self, max_norm, norm_type=2):
        if not self.overflow:
            fp32_params = []
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    fp32_params.append(param)
            return self.clip_grad_norm(fp32_params, max_norm, norm_type)
        else:
            return -1

    def state_dict(self):
        state_dict = {}
        state_dict['loss_scaler'] = self.loss_scaler
        state_dict['dynamic_loss_scale'] = self.dynamic_loss_scale
        state_dict['overflow'] = self.overflow
        state_dict['first_closure_call_this_step'] = self.first_closure_call_this_step
        state_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        state_dict['fp32_from_fp16'] = self.fp32_from_fp16_groups
        return state_dict

    def load_state_dict(self, state_dict):
        self.loss_scaler = state_dict['loss_scaler']
        self.dynamic_loss_scale = state_dict['dynamic_loss_scale']
        self.overflow = state_dict['overflow']
        self.first_closure_call_this_step = state_dict['first_closure_call_this_step']
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        for current_group, saved_group in zip(self.fp32_from_fp16_groups, state_dict['fp32_from_fp16']):
            for current, saved in zip(current_group, saved_group):
                current.data.copy_(saved.data)

    def step(self, closure=None):
        scale = self.loss_scaler.loss_scale()
        if self.overflow:
            maybe_print("Gradient overflow.  Skipping step, reducing loss scale to {}".format(self.loss_scaler.loss_scale()))
            return
        if closure is not None:
            retval = self._step_with_closure(closure)
        else:
            retval = self.optimizer.step()
        self._master_params_to_model_params()
        return retval

    def _step_with_closure(self, closure):
        def wrapped_closure():
            if self.first_closure_call_this_step:
                self.first_closure_call_this_step = False
            else:
                self._master_params_to_model_params()
            temp_loss = closure()
            while self.overflow:
                scale = self.loss_scaler.loss_scale()
                print("OVERFLOW within closure! Skipping step, reducing loss scale to {}".format(
                    self.loss_scaler.loss_scale()))
                temp_loss = closure()
            return temp_loss
        retval = self.optimizer.step(wrapped_closure)
        self.first_closure_call_this_step = True
        return retval

    def backward(self, loss, update_master_grads=True, retain_graph=False):
        scaled_loss = loss.float() * self.loss_scaler.loss_scale()
        scaled_loss.backward(retain_graph=retain_graph)
        if update_master_grads:
            self.update_master_grads()

    def update_master_grads(self):
        self.loss_scaler.clear_overflow_state()
        if len(self.all_fp16_params) > 0:
            model_grads = []
            master_grads = []
            for model_param, master_param in zip(self.all_fp16_params,
                                                 self.all_fp32_from_fp16_params):
                if model_param.grad is not None:
                    model_grads.append(model_param.grad)
                    if master_param.grad is None:
                        master_param.grad = torch.empty_like(master_param)
                    master_grads.append(master_param.grad)
            self.loss_scaler.unscale(
                model_grads,
                master_grads,
                self.loss_scaler.loss_scale())
        if len(self.all_fp32_from_fp32_params) > 0:
            model_grads = []
            master_grads = []
            for model_param, master_param in zip(self.all_fp32_from_fp32_params,
                                                 self.all_fp32_from_fp32_params):
                if model_param.grad is not None:
                    model_grads.append(model_param.grad)
                    master_grads.append(master_param.grad)
            self.loss_scaler.unscale(
                model_grads,
                master_grads,
                self.loss_scaler.loss_scale())
        self.overflow = self.loss_scaler.update_scale()

    def inspect_master_grad_data(self):
        if self.overflow:
            print("Warning:  calling FP16_Optimizer.inspect_master_grad_data while in an overflow state.  Gradients are currently invalid (may be inf, nan, or stale).  Returning None.")
            return None
        else:
            master_grads_data = []
            for param_group in self.optimizer.param_groups:
                master_grads_this_group = []
                for param in param_group['params']:
                    if param.grad is not None:
                        master_grads_this_group.append(param.grad.data)
                    else:
                        master_grads_this_group.append(None)
                master_grads_data.append(master_grads_this_group)
            return master_grads_data

    def _get_loss_scale(self):
        return self.loss_scaler.loss_scale()

    def _set_loss_scale(self, value):
        self.loss_scaler._loss_scale = value

    loss_scale = property(_get_loss_scale, _set_loss_scale)

    def _get_state(self):
        return self.optimizer.state

    def _set_state(self, value):
        self.optimizer.state = value

    state = property(_get_state, _set_state)

    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)