from .weight_norm import WeightNorm
from Code_for_Experiment.Metrics.music_understanding_model.apex.apex.reparameterization.reparameterization import Reparameterization

def apply_weight_norm(module, name='', dim=0, hook_child=True):
    return apply_reparameterization(module, reparameterization=WeightNorm, hook_child=hook_child, name=name, dim=dim)

def remove_weight_norm(module, name='', remove_all=False):
    return remove_reparameterization(module, reparameterization=WeightNorm, name=name, remove_all=remove_all)

def apply_reparameterization(module, reparameterization=None, name='', dim=0, hook_child=True):
    assert reparameterization is not None
    if name != '':
        Reparameterization.apply(module, name, dim, reparameterization, hook_child)
    else:
        names = list(module.state_dict().keys())
        for name in names:
            apply_reparameterization(module, reparameterization, name, dim, hook_child)
    return module

def remove_reparameterization(module, reparameterization=Reparameterization, name='', remove_all=False):
    if name != '' or remove_all:
        to_remove = []
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, reparameterization) and (hook.name == name or remove_all):
                hook.remove(module)
                to_remove.append(k)
        if len(to_remove) > 0:
            for k in to_remove:
                del module._forward_pre_hooks[k]
            return module
        if not remove_all:
            raise ValueError("reparameterization of '{}' not found in {}".format(name, module))
    else:
        modules = [module] + [x for x in module.modules()]
        for m in modules:
            remove_reparameterization(m, reparameterization=reparameterization, remove_all=True)
    return module