import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

class FP16_Optimizer(object):
    def __init__(self, init_optimizer, static_loss_scale=1.0, dynamic_loss_scale=False, dynamic_loss_args=None, verbose=True):
        if not torch.cuda.is_available():
            raise SystemError("Cannot use fp16 without CUDA.")
        self.optimizer = init_optimizer
        self.fp16_groups = []
        self.fp16_groups_flat = []
        self.fp32_groups_flat = []
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.fp16_groups.append(param_group['params'])
            self.fp16_groups_flat.append(_flatten_dense_tensors([p.clone().detach() for p in self.fp16_groups[i]]))
            updated_params = _unflatten_dense_tensors(self.fp16_groups_flat[i], self.fp16_groups[i])
            for p, q in zip(self.fp16_groups[i], updated_params):
                p.data = q.data
            self.fp32_groups_flat.append(self.fp16_groups_flat[i].clone().float().detach())
            self.fp32_groups_flat[i].requires_grad = True
            param_group['params'] = [self.fp32_groups_flat[i]]
        if dynamic_loss_scale:
            if dynamic_loss_args is not None:
                raise SystemError("Do not support dynamic loss scale args for now.")
            self.dynamic_loss_scale = True
            self.cur_scale = 2**16
            self.cur_iter = 0
            self.last_overflow_iter = -1
            self.scale_factor = 2
            self.scale_window = 1000
        else:
            self.dynamic_loss_scale = False
            self.cur_iter = 0
            self.cur_scale = static_loss_scale
        self.verbose = verbose

    def zero_grad(self, set_grads_to_None=True):
        for group in self.fp16_groups:
            for p in group:
                if set_grads_to_None:
                    p.grad = None
                else:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

    def _compute_grad_norm(self, fp16_grads_flat, norm_type=2):
        try:
            norm = float(torch.norm(fp16_grads_flat, 2.0, dtype=torch.float32))
        except TypeError as err:
            norm = float(torch.norm(fp16_grads_flat.float(), 2.0))
        if norm == float('inf') or norm == -float('inf') or norm != norm:
            return -1
        else:
            return norm

    def step(self, closure=None):
        grads_groups_flat = []
        norm_groups = []
        skip = False
        for i, group in enumerate(self.fp16_groups):
            grads_groups_flat.append(_flatten_dense_tensors([p.grad for p in group]))
            norm_groups.append(self._compute_grad_norm(grads_groups_flat[i]))
            if norm_groups[i] == -1:
                skip = True
        if skip:
            self._update_scale(skip)
            return
        self.optimizer.step(grads=[[g] for g in grads_groups_flat],
                            output_params=[[p] for p in self.fp16_groups_flat],
                            scale=self.cur_scale,
                            grad_norms=norm_groups)
        for i in range(len(norm_groups)):
            updated_params = _unflatten_dense_tensors(self.fp16_groups_flat[i], self.fp16_groups[i])
            for p, q in zip(self.fp16_groups[i], updated_params):
                p.data = q.data
        self._update_scale(False)
        return

    def backward(self, loss):
        scaled_loss = (loss.float()) * self.cur_scale
        scaled_loss.backward()

    def _update_scale(self, skip):
        if self.dynamic_loss_scale:
            if skip:
                if self.verbose:
                    print("\nGrad overflow on iteration", self.cur_iter)
                    print("Using dynamic loss scale of", self.cur_scale)
                self.cur_scale = max(self.cur_scale / self.scale_factor, 1)
                self.last_overflow_iter = self.cur_iter
            else:
                if (self.cur_iter - self.last_overflow_iter) % self.scale_window == 0:
                    self.cur_scale *= self.scale_factor
        else:
            if skip:
                print("\nGrad overflow on iteration", self.cur_iter)
                print("Using static loss scale of", self.cur_scale)
        self.cur_iter += 1
        return

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

    def state_dict(self):
        state_dict = {}
        state_dict['dynamic_loss_scale'] = self.dynamic_loss_scale
        state_dict['cur_scale'] = self.cur_scale
        state_dict['cur_iter'] = self.cur_iter
        if state_dict['dynamic_loss_scale']:
            state_dict['last_overflow_iter'] = self.last_overflow_iter
            state_dict['scale_factor'] = self.scale_factor
            state_dict['scale_window'] = self.scale_window
        state_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        state_dict['fp32_groups_flat'] = self.fp32_groups_flat
        return state_dict

    def load_state_dict(self, state_dict):
        self.dynamic_loss_scale = state_dict['dynamic_loss_scale']
        self.cur_scale = state_dict['cur_scale']
        self.cur_iter = state_dict['cur_iter']
        if state_dict['dynamic_loss_scale']:
            self.last_overflow_iter = state_dict['last_overflow_iter']
            self.scale_factor = state_dict['scale_factor']
            self.scale_window = state_dict['scale_window']
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        for current, saved in zip(self.fp32_groups_flat, state_dict['fp32_groups_flat']):
            current.data.copy_(saved.data)