import torch
from torch.nn.parameter import Parameter
import sys

class Reparameterization(object):
    def __init__(self, name, dim, module, retain_forward=True):
        self.name = name
        self.dim = dim
        self.evaluated = False
        self.retain_forward = retain_forward
        self.reparameterization_names = []
        self.backward_hook_key = None
        self.module = module

    def compute_weight(self, module=None, name=None):
        raise NotImplementedError

    def reparameterize(self, name, weight, dim):
        raise NotImplementedError

    @staticmethod
    def apply(module, name, dim, reparameterization=None, hook_child=True):
        if reparameterization is None:
            reparameterization = Reparameterization
        module2use, name2use = Reparameterization.get_module_and_name(module, name)
        if name2use is None or isinstance(module2use, (torch.nn.Embedding, torch.nn.EmbeddingBag)):
            return
        if hook_child:
            fn = reparameterization(name2use, dim, module2use)
        else:
            fn = reparameterization(name, dim, module)
        weight = getattr(module2use, name2use)
        if weight.dim() <= 1:
            return
        del module2use._parameters[name2use]
        names, params = fn.reparameterize(name2use, weight, dim)
        for n, p in zip(names, params):
            module2use.register_parameter(n, p)
        fn.reparameterization_names = names
        setattr(module2use, name2use, None)
        hook_module = module2use
        if not hook_child:
            hook_module = module
        hook_module.register_forward_pre_hook(fn)
        handle = hook_module.register_backward_hook(fn.backward_hook)
        fn.backward_hook_key = handle.id
        return fn

    @staticmethod
    def get_module_and_name(module, name):
        name2use = None
        module2use = None
        names = name.split('.')
        if len(names) == 1 and names[0] != '':
            name2use = names[0]
            module2use = module
        elif len(names) > 1:
            module2use = module
            name2use = names[0]
            for i in range(len(names) - 1):
                module2use = getattr(module2use, name2use)
                name2use = names[i + 1]
        return module2use, name2use

    def get_params(self, module):
        return [getattr(module, n) for n in self.reparameterization_names]

    def remove(self, module):
        module2use, name2use = Reparameterization.get_module_and_name(module, self.name)
        for p in self.get_params(module2use):
            p.requires_grad = False
        weight = self.compute_weight(module2use, name2use)
        delattr(module2use, name2use)
        for n in self.reparameterization_names:
            del module2use._parameters[n]
        module2use.register_parameter(name2use, Parameter(weight.data))
        del module._backward_hooks[self.backward_hook_key]

    def __call__(self, module, inputs):
        module2use, name2use = Reparameterization.get_module_and_name(module, self.name)
        _w = getattr(module2use, name2use)
        if not self.evaluated or _w is None:
            setattr(module2use, name2use, self.compute_weight(module2use, name2use))
            self.evaluated = True

    def backward_hook(self, module, grad_input, grad_output):
        module2use, name2use = Reparameterization.get_module_and_name(module, self.name)
        wn = getattr(module2use, name2use)
        self.evaluated = False