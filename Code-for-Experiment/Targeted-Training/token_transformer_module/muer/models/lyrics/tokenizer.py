from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)
from torch import nn

from ama_prof_divi.utils import *


class LyricsTokenizer(ABC, nn.Module):
    def __init__(self, hparams: dict):
        super(LyricsTokenizer, self).__init__()
        self.hparams = hparams
        self.tok_hparams = self.hparams["ama-prof-divi"]["models"]["lyrics"]["tokenizer"]

        self.pad_token = self.tok_hparams["pad_token"]
        self.start_token = self.tok_hparams["start_token"]
        self.end_token = self.tok_hparams["end_token"]
        self.mask_token = self.tok_hparams["mask_token"]
        self.sep_token = self.tok_hparams["sep_token"]
        self.unknown_token = self.tok_hparams["unknown_token"]

    @property
    def model_name(self) -> str:
        return self._get_model_name()

    @property
    def vocab_size(self) -> int:
        return self._get_vocab_size()

    @abstractmethod
    def _get_model_name(self):
        ...

    @abstractmethod
    def _get_vocab_size(self) -> int:
        ...

    @abstractmethod
    def forward(self, text: str) -> list:
        ...

    @abstractmethod
    def encode(self, text: str) -> list:
        ...

    @abstractmethod
    def encode_batch(self, text: list[str], *, num_threads: int = 8) -> list[list[int]]:
        ...

    @abstractmethod
    def decode(self, token_ids: list, *, errors: str = "replace") -> str:
        ...

    @abstractmethod
    def decode_batch(self, tokens: list[list[int]], *, errors: str = "replace", num_threads: int = 8) -> list[str]:
        ...

    @abstractmethod
    def special_tokens_set(self):
        ...

    @abstractmethod
    def is_special_token(self, token: str or int) -> bool:
        ...

    @abstractmethod
    def encode_special_token(self, token: str) -> int:
        ...

    def pad_id(self) -> int:
        return self.encode_special_token(self.pad_token)

    def start_id(self) -> int:
        return self.encode_special_token(self.start_token)

    def end_id(self) -> int:
        return self.encode_special_token(self.end_token)

    def mask_id(self) -> int:
        return self.encode_special_token(self.mask_token)

    def sep_id(self) -> int:
        return self.encode_special_token(self.sep_token)

    def unknown_id(self) -> int:
        return self.encode_special_token(self.unknown_token)