from abc import ABC, abstractmethod
import torch
import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.amp.rnn_compat
from typing import List, Tuple

FeatureMapType = List[torch.Tensor]
LogitsType = torch.Tensor
MultiDiscriminatorOutputType = Tuple[List[LogitsType], List[FeatureMapType]]

class MultiDiscriminator(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> MultiDiscriminatorOutputType:
        pass

    @property
    @abstractmethod
    def num_discriminators(self) -> int:
        pass