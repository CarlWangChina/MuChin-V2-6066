from abc import ABC, abstractmethod
import torch
import transformers
from music_dit.utils import get_logger, get_hparams

logger = get_logger(__name__)

class Translator(ABC):
    def __init__(self):
        super(Translator, self).__init__()
        self.hparams = get_hparams()

    @property
    def checkpoint(self) -> str:
        return self._get_checkpoint()

    @abstractmethod
    def _get_checkpoint(self) -> str:
        ...

    @abstractmethod
    def translate(self, text: str) -> str:
        ...

class TranslatorZhToEn(Translator):
    def __init__(self):
        super(TranslatorZhToEn, self).__init__()
        self.translator_hparams = self.hparams.clap.translators.zh_en
        self.pipeline = transformers.pipeline("translation", model=self._get_checkpoint(), device=self.hparams.device)

    def _get_checkpoint(self) -> str:
        return self.translator_hparams.pretrained_model

    @torch.no_grad()
    def translate(self, text: str) -> str:
        return self.pipeline(text)[0]["translation_text"]

class TranslatorEnToZh(Translator):
    def __init__(self):
        super(TranslatorEnToZh, self).__init__()
        self.translator_hparams = self.hparams.clap.translators.en_zh
        self.pipeline = transformers.pipeline("translation", model=self._get_checkpoint(), device=self.hparams.device)

    def _get_checkpoint(self) -> str:
        return self.translator_hparams.pretrained_model

    @torch.no_grad()
    def translate(self, text: str) -> str:
        return self.pipeline(text)[0]["translation_text"]