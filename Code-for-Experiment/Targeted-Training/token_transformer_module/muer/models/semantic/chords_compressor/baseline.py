import torch
import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.amp.rnn_compat
from einops import rearrange
import torch.nn as nn

class ChordsCompressorBaseline(nn.Module):
    def __init__(self, hparams: dict):
        super(ChordsCompressorBaseline, self).__init__()
        self.hparams = hparams
        self.comp_hparams = self.hparams["ama-prof-divi"]["models"]["semantic"]["chords_compressor"]
        device = self.hparams["ama-prof-divi"]["device"]
        self.dim = self.comp_hparams["n_dim"]
        self.compress_ratio = self.comp_hparams["compress_ratio"]
        self.pool = nn.AvgPool1d(kernel_size=self.compress_ratio, stride=self.compress_ratio).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, "The input tensor should be 3D."
        assert x.size(-1) == self.dim, "The input tensor should have the same dimension as the model."
        x = rearrange(x, "b t d -> b d t")
        x = self.pool(x)
        x = rearrange(x, "b d t -> b t d")
        return x

class ChordsDecompressorBaseline(nn.Module):
    def __init__(self, hparams: dict):
        super(ChordsDecompressorBaseline, self).__init__()
        self.hparams = hparams
        self.comp_hparams = self.hparams["ama-prof-divi"]["models"]["semantic"]["chords_compressor"]
        self.dim = self.comp_hparams["n_dim"]
        self.compress_ratio = self.comp_hparams["compress_ratio"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, "The input tensor should be 3D."
        assert x.size(-1) == self.dim, "The input tensor should have the same dimension as the model."
        return torch.repeat_interleave(x, self.compress_ratio, dim=1)