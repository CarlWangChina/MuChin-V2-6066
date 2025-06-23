import torch
import torch.nn as nn
from ama_prof_divi.utils.logging import get_logger
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.models.semantic.chords_compressor.baseline import ChordsDecompressorBaseline

logger = get_logger(__name__)

class ChordsDecompressor(nn.Module):
    def __init__(self, hparams: dict):
        super(ChordsDecompressor, self).__init__()
        self.hparams = hparams
        self.model = ChordsDecompressorBaseline(self.hparams)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @property
    def dim(self):
        return self.model.dim

    @property
    def compress_ratio(self):
        return self.model.compress_ratio