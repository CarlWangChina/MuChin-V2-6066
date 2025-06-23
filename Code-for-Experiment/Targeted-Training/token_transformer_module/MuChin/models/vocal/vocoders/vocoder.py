import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import os
import zipfile
import json
from ama_prof_divi.utils import download_file
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.logging import get_logger
logger = get_logger(__name__)
import HifiGanGenerator
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.models.vocal.vocoders.mel_encoder import MelEncoder

def _download_pretrained_model(checkpoints_path: Path, model_name: str, pretrained_model_url: str, pretrained_model_sha256: str):
    temp_dir = tempfile.mkdtemp()
    logger.info("Temp dir: {}".format(temp_dir))
    compressed_file = Path(temp_dir) / (model_name + "-latest.zip")
    download_file(pretrained_model_url, str(compressed_file), pretrained_model_sha256)
    with zipfile.ZipFile(compressed_file, "r") as zip_file:
        zip_file.extractall(checkpoints_path)
    os.unlink(compressed_file)
    os.rmdir(temp_dir)

def _create_model_from_pretrained(model_path: Path, vocoder_hparams: dict, device: str) -> (nn.Module, dict):
    assert model_path.exists() and model_path.is_dir(), f"Model {model_path} does not exist."
    config_path = model_path / "config.json"
    assert config_path.exists() and config_path.is_file(), f"Config {config_path} does not exist."
    with open(config_path, "r") as f:
        config = json.load(f)
    model = HifiGanGenerator(sampling_rate=config["sampling_rate"],
                             num_mel_bands=config["num_mels"],
                             up_sampling_rates=config["upsample_rates"],
                             up_sampling_kernel_sizes=config["upsample_kernel_sizes"],
                             up_sampling_initial_channels=config["upsample_initial_channel"],
                             res_block_kernel_sizes=config["resblock_kernel_sizes"],
                             res_block_type=config["resblock"],
                             res_block_dilation_sizes=config["resblock_dilation_sizes"],
                             use_pitch_embedding=vocoder_hparams["use_pitch_embedding"],
                             harmonic_num=vocoder_hparams["harmonic_num"],
                             device=device)
    states_path = model_path / "model"
    assert states_path.exists() and states_path.is_file(), f"States {states_path} does not exist."
    state_dict = torch.load(states_path, map_location=device)["generator"]
    model.load_state_dict(state_dict, strict=True)
    return model, config

class SVSVocoder(nn.Module):
    def __init__(self, hparams: dict):
        super(SVSVocoder, self).__init__()
        self.vocoder_hparams = hparams["ama-prof-divi"]["models"]["vocal"]["vocoder"]
        self.device = hparams["ama-prof-divi"]["device"]
        assert self.vocoder_hparams["name"] == "hifi-gan", f"Unsupported vocoder: {self.vocoder_hparams['name']}"
        root_path = Path(hparams["ama-prof-divi"]["root_path"])
        checkpoints_path = root_path / "checkpoints"
        model_path = checkpoints_path / self.vocoder_hparams["model_name"]
        if not (model_path.exists() and model_path.is_dir()):
            logger.info(f"Model {self.vocoder_hparams['model_name']} does not exist.  Loading pretrained model...")
            _download_pretrained_model(checkpoints_path,
                                       self.vocoder_hparams["model_name"],
                                       self.vocoder_hparams["pretrained_model_url"],
                                       self.vocoder_hparams["pretrained_model_sha256"])
        self.generator, self.configs = _create_model_from_pretrained(model_path,
                                                                     self.vocoder_hparams,
                                                                     self.device)
        logger.info(f"Loaded pretrained SVS Hifi-GAN model: {self.vocoder_hparams['model_name']}")
        self.mel_encoder = MelEncoder(self.configs, device=self.device)

    def forward(self, mels: torch.Tensor, f0: torch.Tensor = None) -> torch.Tensor:
        return self.generator(mels, f0)

    def mel_encode(self, audio: torch.Tensor, center: bool = False) -> torch.Tensor:
        return self.mel_encoder(audio, center)

    @property
    def sampling_rate(self) -> int:
        return self.configs["sampling_rate"]

    @property
    def num_mel_bands(self) -> int:
        return self.configs["num_mels"]

    @property
    def num_fft(self) -> int:
        return (self.num_mel_bands - 1) * 2

    @property
    def f_min(self) -> int:
        return self.configs["fmin"]

    @property
    def f_max(self) -> int:
        return self.configs["fmax"]

    @property
    def hop_size(self) -> int:
        return self.configs["hop_size"]