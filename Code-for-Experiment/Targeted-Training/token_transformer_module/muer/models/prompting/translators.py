from abc import ABC, abstractmethod
import transformers
import model_need
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger

logger = get_logger(__name__)

class Translator(ABC):
    def __init__(self, hparams: dict):
        super(Translator, self).__init__()
        self.hparams = hparams
        self.translator_hparams = self.hparams["ama-prof-divi"]["models"]["prompting"]["translator"]
        self.device = "cpu"

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
    def __init__(self, hparams: dict):
        super(TranslatorZhToEn, self).__init__(hparams)
        self.hparams = hparams
        self.translator_hparams = self.translator_hparams["zh_en"]
        self.pipeline = transformers.pipeline("translation", model=self._get_checkpoint(), device=self.device)
        if model_need.freeze(__name__):
            logger.info("The zh-en translator is set to freeze.")
            model_need.freeze_model(self.pipeline.model)

    def _get_checkpoint(self) -> str:
        return self.translator_hparams["pretrained_model"]

    def translate(self, text: str) -> str:
        return self.pipeline(text)[0]["translation_text"]

class TranslatorEnToZh(Translator):
    def __init__(self, hparams: dict):
        super(TranslatorEnToZh, self).__init__(hparams)
        self.hparams = hparams
        self.translator_hparams = self.translator_hparams["en_zh"]
        self.pipeline = transformers.pipeline("translation", model=self._get_checkpoint(), device=self.device)
        if model_need.freeze(__name__):
            logger.info("The en-zh translator is set to freeze.")
            model_need.freeze_model(self.pipeline.model)

    def _get_checkpoint(self) -> str:
        return self.translator_hparams["checkpoint"]

    def translate(self, text: str) -> str:
        return self.pipeline(text)[0]["translation_text"]