from abc import ABC, abstractmethod
import torch.nn as nn
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger

logger = get_logger(__name__)
DEFAULT_PROMPT = "A pop song with a happy mood."

class PromptEncoder(ABC, nn.Module):
    def __init__(self, hparams: dict):
        super(PromptEncoder, self).__init__()
        self.hparams = hparams
        self.encoder_hparams = self.hparams["ama-prof-divi"]["models"]["prompting"]["encoder"]

    @property
    def model_name(self) -> str:
        return self._get_model_name()

    @property
    def checkpoint(self) -> str:
        return self._get_checkpoint()

    @property
    def sampling_rate(self) -> int:
        return self._get_sampling_rate()

    @property
    def max_clip_samples(self) -> int:
        return self._get_max_clip_samples()

    @property
    def vocab_size(self) -> int:
        return self._get_vocab_size()

    @property
    def joint_embedding_dim(self) -> int:
        return self._get_joint_embedding_dim()

    @property
    def languages(self) -> [str]:
        return self._get_languages()

    @property
    def device(self) -> torch.device:
        return self._get_device()

    @abstractmethod
    def _get_device(self) -> torch.device:
        ...

    @abstractmethod
    def _get_model_name(self) -> str:
        ...

    @abstractmethod
    def _get_checkpoint(self) -> str:
        ...

    @abstractmethod
    def _get_sampling_rate(self) -> int:
        ...

    @abstractmethod
    def _get_max_clip_samples(self) -> int:
        ...

    @abstractmethod
    def _get_vocab_size(self) -> int:
        ...

    @abstractmethod
    def _get_joint_embedding_dim(self) -> int:
        ...

    @abstractmethod
    def get_text_embedding(self, text: [str]) -> torch.Tensor:
        ...

    @abstractmethod
    def get_text_embedding_zh(self, text: [str]) -> torch.Tensor:
        ...

    @abstractmethod
    def get_audio_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def _get_languages(self) -> [str]:
        ...

    def get_text_prompt_embeddings(self, *, text_prompt: [str] = None, text_prompt_language: str = "en", text_prompt_embedding: torch.Tensor = None) -> (torch.Tensor, int):
        if text_prompt_embedding is not None:
            if text_prompt is not None:
                logger.warning("text_prompt_embedding is Used.  text_prompt will be ignored.")
            assert text_prompt_embedding.dim() == 2
            assert text_prompt_embedding.shape[1] == self.joint_embedding_dim
            num_batches = text_prompt_embedding.shape[0]
        else:
            assert text_prompt_language in self.languages, \
                (f"Language {text_prompt_language} is not supported.  Should be one of {self.languages}.")
            num_batches = 0
            if text_prompt is None:
                text_prompt = [DEFAULT_PROMPT]
                text_prompt_embedding = self.get_text_embedding(text_prompt)
            else:
                num_batches = len(text_prompt)
                text_prompt_embedding = self.get_text_embedding_zh(text_prompt) \
                    if text_prompt_language == "zh" else self.get_text_embedding(text_prompt)
        return text_prompt_embedding, num_batches