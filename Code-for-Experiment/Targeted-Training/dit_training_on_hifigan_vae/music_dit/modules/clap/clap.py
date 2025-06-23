import torch
import torch.nn as nn
import laion_clap
from omegaconf import DictConfig
from music_dit.utils import get_logger, download_file, get_hparams
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.modules.translators.translators import TranslatorZhToEn

logger = get_logger(__name__)

class ClapEncoder(nn.Module):
    def __init__(self):
        super(ClapEncoder, self).__init__()
        hparams = get_hparams()
        device = hparams.device
        self.translator_zh_en = TranslatorZhToEn()
        self.clap_model = laion_clap.CLAP_Module(enable_fusion=hparams.clap.enable_fusion,
                                                 amodel=hparams.clap.amodel,
                                                 tmodel=hparams.clap.tmodel,
                                                 device=device)
        self._load_pretrained_checkpoints(hparams)

    def _load_pretrained_checkpoints(self,
                                     hparams: DictConfig):
        checkpoints_dir = hparams.checkpoint_dir / "laion-clap"
        if not checkpoints_dir.exists():
            logger.info(f"Creating directory '{checkpoints_dir}' ...")
            checkpoints_dir.mkdir(parents=False)
        checkpoints_file = checkpoints_dir.joinpath(hparams.clap.pretrained_model)
        logger.info(f"Downloading pretrained checkpoints '{hparams.clap.pretrained_model}' ...")
        checksum = None
        if "pretrained_model_sha256" in hparams.clap:
            checksum = hparams.clap.pretrained_model_sha256
        download_file(hparams.clap.pretrained_model_url,
                      str(checkpoints_file),
                      expected_sha256=checksum)
        logger.info(f"Loading pretrained checkpoints '{hparams.clap.pretrained_model}' ...")
        self.clap_model.load_ckpt(str(checkpoints_file))

    @property
    def sampling_rate(self):
        return self.clap_model.model_cfg["audio_cfg"]["sample_rate"]

    @property
    def num_channels(self):
        return 1

    @property
    def joint_embedding_dim(self) -> int:
        return self.clap_model.model_cfg["text_cfg"]["width"]

    @property
    def max_clip_samples(self) -> int:
        return self.clap_model.model_cfg["audio_cfg"]["clip_samples"]

    @torch.no_grad()
    def get_text_embedding(self, text: [str]) -> torch.Tensor:
        if len(text) == 0:
            return torch.tensor([])
        if len(text) == 1:
            text = [text[0], ""]
            return self.clap_model.get_text_embedding(text, use_tensor=True).detach()[:1, :]
        return self.clap_model.get_text_embedding(text, use_tensor=True).detach()

    @torch.no_grad()
    def get_text_embedding_zh(self, text: [str]) -> torch.Tensor:
        translated_text = [self.translator_zh_en.translate(t) for t in text]
        return self.get_text_embedding(translated_text)

    @torch.no_grad()
    def get_audio_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        assert audio.ndim == 2, f"Audio must be 2D tensor, got {audio.ndim}D tensor."
        return self.clap_model.get_audio_embedding_from_data(audio, use_tensor=True)