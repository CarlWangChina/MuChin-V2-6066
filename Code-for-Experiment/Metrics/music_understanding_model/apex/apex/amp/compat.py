import torch

def variable_is_tensor():
    v = torch.autograd.Variable()
    return isinstance(v, torch.Tensor)

def tensor_is_variable():
    x = torch.Tensor()
    return type(x) == torch.autograd.Variable

def tensor_is_float_tensor():
    x = torch.Tensor()
    return type(x) == torch.FloatTensor

def is_tensor_like(x):
    return torch.is_tensor(x) or isinstance(x, torch.autograd.Variable)

def is_floating_point(x):
    if hasattr(torch, 'is_floating_point'):
        return torch.is_floating_point(x)
    try:
        torch_type = x.type()
        return torch_type.endswith('FloatTensor') or \
            torch_type.endswith('HalfTensor') or \
            torch_type.endswith('DoubleTensor')
    except AttributeError:
        return False

def scalar_python_val(x):
    if hasattr(x, 'item'):
        return x.item()
    else:
        if isinstance(x, torch.autograd.Variable):
            return x.data[0]
        else:
            return x[0]