import math
import torch
import numbers
from torch.nn.parameter import Parameter
from Code_for_Experiment.Targeted_Training.data_prep_demucs.demucs.distrib import init
from torch.nn import functional as F
import importlib

class FusedLayerNormAffineFunction(torch.autograd.Function):
    def __init__(self, normalized_shape, eps=1e-6):
        global fused_layer_norm_cuda
        fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, input, weight, bias):
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        bias_ = bias.contiguous()
        output, mean, invvar = fused_layer_norm_cuda.forward_affine(
            input_, self.normalized_shape, weight_, bias_, self.eps)
        self.save_for_backward(input_, weight_, bias_, mean, invvar)
        return output

    def backward(self, grad_output):
        input_, weight_, bias_, mean, invvar = self.saved_tensors
        grad_input = grad_weight = grad_bias = None
        grad_input, grad_weight, grad_bias = fused_layer_norm_cuda.backward_affine(
            grad_output.contiguous(), mean, invvar,
            input_, self.normalized_shape,
            weight_, bias_, self.eps)
        return grad_input, grad_weight, grad_bias

class FusedLayerNormFunction(torch.autograd.Function):
    def __init__(self, normalized_shape, eps=1e-6):
        global fused_layer_norm_cuda
        fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, input):
        input_ = input.contiguous()
        output, mean, invvar = fused_layer_norm_cuda.forward(
            input_, self.normalized_shape, self.eps)
        self.save_for_backward(input_, mean, invvar)
        return output

    def backward(self, grad_output):
        input_, mean, invvar = self.saved_tensors
        grad_input = None
        grad_input = fused_layer_norm_cuda.backward(
            grad_output.contiguous(), mean, invvar,
            input_, self.normalized_shape,
            self.eps)
        return grad_input

def fused_layer_norm_affine(input, normalized_shape, weight, bias, eps=1e-6):
    return FusedLayerNormAffineFunction(normalized_shape, eps)(input, weight, bias)

def fused_layer_norm(input, normalized_shape, eps=1e-6):
    return FusedLayerNormFunction(normalized_shape, eps)(input)

class FusedLayerNorm(torch.nn.Module):
    r"""
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(FusedLayerNorm, self).__init__()
        global fused_layer_norm_cuda
        fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):
        if not input.is_cuda:
            return F.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps)
        if self.elementwise_affine:
            return FusedLayerNormAffineFunction(self.normalized_shape, self.eps)(
                input, self.weight, self.bias)
        else:
            return FusedLayerNormFunction(self.normalized_shape, self.eps)(
                input)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)