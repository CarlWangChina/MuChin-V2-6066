import torch
from .. import utils

MODULE = torch

FP16_FUNCS = [
    'conv1d',
    'conv2d',
    'conv3d',
    'conv_transpose1d',
    'conv_transpose2d',
    'conv_transpose3d',
    'conv_tbc',
    'prelu',
    'addmm',
    'addmv',
    'addr',
    'matmul',
    'mm',
    'mv',
]

FP32_FUNCS = [
    'acos',
    'asin',
    'cosh',
    'erfinv',
    'exp',
    'expm1',
    'log',
    'log10',
    'log2',
    'reciprocal',
    'rsqrt',
    'sinh',
    'tan',
    'pow',
    'cumprod',
    'cumsum',
    'dist',
    'mean',
    'norm',
    'prod',
    'std',
    'sum',
    'var',
    'renorm',
]

_bmms = [
    'addbmm',
    'baddbmm',
    'bmm',
]

if utils.get_cuda_version() >= (9, 1, 0):
    FP16_FUNCS.extend(_bmms)
else:
    FP32_FUNCS.extend(_bmms)

CASTS = [
    'addcdiv',
    'addcmul',
    'atan2',
    'cross',
    'bilinear',
    'add',
    'div',
    'mul',
    'eq',
    'equal',
    'ge',
    'gt',
    'le',
    'lt',
    'ne',
]

SEQUENCE_CASTS = [
    'cat',
    'stack',
]