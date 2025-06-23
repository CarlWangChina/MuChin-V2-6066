import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger

logger = get_logger(__name__)

def _unfold(a: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    *shape, length = a.shape
    n_frames = math.ceil(length / stride)
    tgt_length = (n_frames - 1) * stride + kernel_size
    a = F.pad(a, (0, tgt_length - length))
    strides = list(a.stride())
    assert strides[-1] == 1, "data should be contiguous"
    strides = strides[:-1] + [stride, 1]
    return a.as_strided([*shape, n_frames, kernel_size], strides)

def _center(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(-1, True)

def _norm2(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).sum(-1, True)

class SISNRLoss(nn.Module):
    def __init__(self, sample_rate: int = 16000, segment: Optional[float] = 20, overlap: float = 0.5, epsilon: float = torch.finfo(torch.float32).eps):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment = segment
        self.overlap = overlap
        self.epsilon = epsilon

    def forward(self, out_sig: torch.Tensor, ref_sig: torch.Tensor) -> torch.Tensor:
        B, C, T = ref_sig.shape
        assert ref_sig.shape == out_sig.shape
        if self.segment is None:
            frame = T
            stride = T
        else:
            frame = int(self.segment * self.sample_rate)
            stride = int(frame * (1 - self.overlap))
        epsilon = self.epsilon * frame
        gt = _unfold(ref_sig, frame, stride)
        est = _unfold(out_sig, frame, stride)
        if self.segment is None:
            assert gt.shape[-1] == 1
        gt = _center(gt)
        est = _center(est)
        dot = torch.einsum("b c f t, b c f t -> b c f", gt, est)
        proj = dot[:, :, :, None] * gt / (epsilon + _norm2(gt))
        noise = est - proj
        sisnr = 10 * (torch.log10(epsilon + _norm2(proj)) - torch.log10(epsilon + _norm2(noise)))
        return -1 * sisnr[..., 0].mean() / 50.0