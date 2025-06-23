import torch
import torch.nn as nn
from ama_prof_divi.utils.logging import get_logger
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.models.semantic.chords_compressor.baseline import ChordsCompressorBaseline

logger = get_logger(__name__)

class ChordsCompressor(nn.Module):
    def __init__(self, hparams: dict):
        super(ChordsCompressor, self).__init__()
        self.hparams = hparams
        self.model = ChordsCompressorBaseline(self.hparams)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @property
    def dim(self):
        return self.model.dim

    @property
    def compress_ratio(self):
        return self.model.compress_ratio