import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class LARC(object):
    def __init__(self, optimizer, trust_coefficient=0.02, clip=True, eps=1e-8):
        self.param_groups = optimizer.param_groups
        self.optim = optimizer
        self.trust_coefficient = trust_coefficient
        self.eps = eps
        self.clip = clip

    def __getstate__(self):
        return self.optim.__getstate__()

    def __setstate__(self, state):
        self.optim.__setstate__(state)

    def __repr__(self):
        return self.optim.__repr__()

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)

    def step(self):
        with torch.no_grad():
            weight_decays = []
            for group in self.optim.param_groups:
                weight_decay = group['weight_decay'] if 'weight_decay' in group else 0
                weight_decays.append(weight_decay)
                group['weight_decay'] = 0
                for p in group['params']:
                    if p.grad is None:
                        continue
                    param_norm = torch.norm(p.data)
                    grad_norm = torch.norm(p.grad.data)
                    if param_norm != 0 and grad_norm != 0:
                        adaptive_lr = self.trust_coefficient * (param_norm) / (grad_norm + param_norm * weight_decay + self.eps)
                        if self.clip:
                            adaptive_lr = min(adaptive_lr / group['lr'], 1)
                        p.grad.data += weight_decay * p.data
                        p.grad.data *= adaptive_lr
        self.optim.step()
        for i, group in enumerate(self.optim.param_groups):
            group['weight_decay'] = weight_decays[i]