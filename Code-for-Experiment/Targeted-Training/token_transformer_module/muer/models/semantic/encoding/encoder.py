from abc import ABC, abstractmethod
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class SemanticEncoder(ABC, nn.Module):
    def __init__(self, hparams: dict):
        super(SemanticEncoder, self).__init__()
        self.hparams = hparams
        self.enc_hparams = self.hparams["ama-prof-divi"]["models"]["semantic"]["encoder"]

    @property
    def model_name(self) -> str:
        return self._get_model_name()

    @property
    def features_rate(self) -> int:
        return self._get_features_rate()

    @property
    def window_size(self) -> int:
        return self._get_window_size()

    @property
    def sampling_rate(self) -> int:
        return self._get_sampling_rate()

    @property
    def num_channels(self) -> int:
        return self._get_num_channels()

    @property
    def features_dim(self) -> int:
        return self._get_features_dim()

    @abstractmethod
    def _get_model_name(self) -> str:
        ...

    @abstractmethod
    def _get_features_rate(self) -> int:
        ...

    @abstractmethod
    def _get_window_size(self) -> int:
        ...

    @abstractmethod
    def _get_sampling_rate(self) -> int:
        ...

    @abstractmethod
    def _get_num_channels(self) -> int:
        ...

    @abstractmethod
    def _get_features_dim(self) -> int:
        ...

    @abstractmethod
    def encode(self, audio: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        ...

    @abstractmethod
    def forward(self, audio: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        ...