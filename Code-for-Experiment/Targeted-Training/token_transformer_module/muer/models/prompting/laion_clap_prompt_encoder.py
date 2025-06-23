import os
import torch
import laion_clap
from pathlib import Path
from typing import Optional
from ama_prof_divi.utils import download_file
from ama_prof_divi.utils.logging import get_logger
from .prompt_encoder import PromptEncoder
from .translator_zh_to_en import TranslatorZhToEn
from ama_prof_divi.utils.training import model_need_freeze, freeze_model
logger = get_logger(__name__)

class LaionClapPromptEncoder(PromptEncoder):
    def __init__(self, hparams: dict, device: Optional[str or torch.device] = None):
        super(LaionClapPromptEncoder, self).__init__(hparams)
        logger.info("Initializing Laion-Clap prompt encoder...")
        if device is None:
            device = self.hparams["ama-prof-divi"]["device"]
        self.clap_model = laion_clap.CLAP_Module(enable_fusion=self.encoder_hparams["enable_fusion"],
                                                 amodel=self.encoder_hparams["amodel"],
                                                 tmodel=self.encoder_hparams["tmodel"],
                                                 device=device)
        self._load_pretrained_checkpoints()
        self.translator_zh_en = TranslatorZhToEn(hparams)
        if model_need_freeze(__name__):
            logger.info("The Laion-Clap model is set to freeze.")
            freeze_model(self.clap_model)

    def _load_pretrained_checkpoints(self):
        root_path = Path(self.hparams["ama-prof-divi"]["root_path"])
        checkpoints_dir = root_path.joinpath("checkpoints").joinpath(self._get_model_name())
        if not os.path.exists(checkpoints_dir):
            logger.info(f"Creating directory '{checkpoints_dir}' ...")
            os.makedirs(checkpoints_dir)
        checkpoints_file = checkpoints_dir.joinpath(self.encoder_hparams["pretrained_model"])
        logger.info(f"Downloading pretrained checkpoints '{self.encoder_hparams['pretrained_model']}' ...")
        checksum = None
        if "pretrained_model_sha256" in self.encoder_hparams:
            checksum = self.encoder_hparams["pretrained_model_sha256"]
        download_file(self.encoder_hparams["pretrained_model_url"],
                      str(checkpoints_file),
                      expected_sha256=checksum)
        logger.info(f"Loading pretrained checkpoints '{self.encoder_hparams['pretrained_model']}' ...")
        self.clap_model.load_ckpt(str(checkpoints_file))

    def _get_model_name(self):
        return self.encoder_hparams["name"]

    def _get_checkpoint(self) -> str:
        return self.encoder_hparams["pretrained_model"]

    def _get_sampling_rate(self) -> int:
        return self.clap_model.model_cfg["audio_cfg"]["sample_rate"]

    def _get_max_clip_samples(self) -> int:
        return self.clap_model.model_cfg["audio_cfg"]["clip_samples"]

    def _get_vocab_size(self) -> int:
        return self.clap_model.model_cfg["text_cfg"]["vocab_size"]

    def _get_joint_embedding_dim(self) -> int:
        return self.clap_model.model_cfg["text_cfg"]["width"]

    @torch.no_grad()
    def get_text_embedding(self, text: [str]) -> torch.Tensor:
        if len(text) == 0:
            return torch.tensor([])
        if len(text) == 1:
            text = [text[0], ""]
        return self.clap_model.get_text_embedding(text, use_tensor=True).detach()[:1, :]

    @torch.no_grad()
    def get_text_embedding_zh(self, text: [str]) -> torch.Tensor:
        translated_text = [self.translator_zh_en.translate(t) for t in text]
        return self.get_text_embedding(translated_text)

    @torch.no_grad()
    def get_audio_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        assert audio.ndim == 2, f"Audio must be 2D tensor, got {audio.ndim}D tensor."
        return self.clap_model.get_audio_embedding_from_data(audio, use_tensor=True)

    def _get_languages(self) -> [str]:
        return ["en", "zh"]

    def _get_device(self) -> torch.device:
        return next(self.clap_model.model.parameters()).device