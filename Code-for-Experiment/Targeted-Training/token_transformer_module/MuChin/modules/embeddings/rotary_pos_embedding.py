import torch
import torch.nn as nn
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger

logger = get_logger(__name__)

def _precompute_theta_pos_frequencies(dim: int, max_seq_len: int, theta: float = 10000.0, device: str or torch.device = 'cpu') -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs).to(device)
    return freqs_cis

class RotaryPosEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, theta: float = 10000.0, device: str or torch.device = 'cpu'):
        super(RotaryPosEmbedding, self).__init__()
        if torch.device(device).type == 'mps':
            logger.warning("ComplexFloat is not supported by the mps backend.  Falling back to CPU.")
            self.mps_special = True
            device = 'cpu'
        else:
            self.mps_special = False
        self.max_seq_len = max_seq_len
        freqs_cis = _precompute_theta_pos_frequencies(dim, max_seq_len, theta, device)
        self.register_buffer("freqs_cis", freqs_cis)

    @property
    def device(self):
        return self.freqs_cis.device

    @torch.no_grad()
    def forward(self, x: torch.Tensor, start_pos: int = 0, pos_bias: int or list or torch.Tensor = 0) -> torch.Tensor:
        if not torch.is_tensor(pos_bias):
            pos_bias = torch.tensor(pos_bias, dtype=torch.long, device=self.device)
        assert pos_bias.dim() == 0 or pos_bias.dim() == 1
        dev = x.device
        if self.mps_special:
            x = x.to(self.device)
        seq_len = x.shape[1]
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        if pos_bias.dim() > 0:
            freqs_cis = torch.zeros((x.shape[0], seq_len, x.shape[-1]), dtype=x.dtype, device=self.device)
            for i in range(x.shape[0]):
                assert start_pos + pos_bias[i] + seq_len <= self.max_seq_len, \
                    f'Position {start_pos + pos_bias[i] + seq_len} is too large. '
                freqs_cis[i] = self.freqs_cis[start_pos + pos_bias[i]:start_pos + pos_bias[i] + seq_len, :]
        else:
            assert start_pos + pos_bias + seq_len <= self.max_seq_len, \
                f'Position {start_pos + pos_bias + seq_len} is too large. '
            freqs_cis = self.freqs_cis[start_pos + pos_bias:start_pos + pos_bias + seq_len, :]
            freqs_cis = freqs_cis.unsqueeze(0).expand(x.shape[0], -1, -1)
        assert freqs_cis.shape == (x.shape[0], seq_len, x.shape[-1]), \
            (f'Frequency tensor has wrong shape. Should be ({seq_len}, {x.shape[-1]}), actual is ({freqs_cis.shape}).\n'
             f'Maximum shape is {self.freqs_cis.shape}. ')
        freqs_cis = freqs_cis.view(x.shape)
        x = torch.view_as_real(x * freqs_cis).flatten(-2)
        if self.mps_special:
            x = x.to(dev)
        return x

    def embed_on_transformer(self, x: torch.Tensor, start_pos: int = 0, pos_bias: int or list or torch.Tensor = 0) -> torch.Tensor:
        return self.forward(x, start_pos, pos_bias)