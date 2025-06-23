import logging
import torch
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.audioutils import get_audio_utils
from ama_prof_divi.utils import logging
from .encoder import SemanticEncoder

logger = logging.get_logger(__name__)

class MertEncoder(SemanticEncoder):
    def __init__(self, hparams: dict):
        super(MertEncoder, self).__init__(hparams)
        self.device = self.hparams["ama-prof-divi"]["device"]

        logger.info(f"Building MERT encoder from pre-trained model {self.enc_hparams['pretrained_model']}.")
        self.model = AutoModel.from_pretrained(self.enc_hparams["pretrained_model"], trust_remote_code=True).to(self.device)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.enc_hparams["pretrained_model"], trust_remote_code=True)
        logger.info(f"Feature dim = {self.features_dim}, feature rate = {self.features_rate} Hz.")
        self.padding = self.sampling_rate // self.features_rate
        self.audio_utils = get_audio_utils()

    def _get_model_name(self):
        return "mert:" + self.model.name_or_path

    def _get_features_rate(self):
        return self.enc_hparams["features_rate"]

    def _get_window_size(self):
        return self.enc_hparams["window_size"]

    def _get_sampling_rate(self) -> int:
        return self.feature_extractor.sampling_rate

    def _get_num_channels(self) -> int:
        return self.enc_hparams["num_channels"]

    def _get_features_dim(self) -> int:
        return self.enc_hparams["features_dim"]

    def encode(self, audio: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        window_size_in_samples = self.window_size * self.feature_extractor.sampling_rate
        window_size_in_features = self.window_size * self.features_rate
        audio = self.audio_utils.resample(audio, sampling_rate, self.sampling_rate, 1)
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
        result = torch.Tensor()
        result_num_features = audio.shape[-1] * self.features_rate // self.sampling_rate
        for batch in range(audio.shape[0]):
            audio_batch = audio[batch]
            assert audio_batch.dim() == 2 and audio_batch.shape[0] == 1
            audio_batch = audio_batch[0]
            output = torch.Tensor()
            for i in range(0, audio_batch.shape[0], window_size_in_samples):
                audio_window = audio_batch[i: i + window_size_in_samples + self.padding]
                if audio_window.shape[0] < window_size_in_samples + self.padding:
                    audio_window = torch.cat((audio_window, torch.zeros(window_size_in_samples + self.padding - audio_window.shape[0])))
                features = self.feature_extractor(audio_window, sampling_rate=self.sampling_rate, padding=True, return_attention_mask=True, return_tensors="pt")
                features["input_values"] = features["input_values"].to(self.device)
                features["attention_mask"] = features["attention_mask"].to(self.device)
                features = self.model(**features)["last_hidden_state"].to("cpu")
                assert features.dim() == 3 and features.shape[0] == 1 and features.shape[-1] == self.features_dim
                assert features.shape[1] >= window_size_in_features, f"features.shape[1] = {features.shape[1]}, window_size_in_features = {window_size_in_features}"
                if i + window_size_in_samples >= audio_batch.shape[0]:
                    ws = (audio_batch.shape[0] - i) * self.features_rate // self.sampling_rate
                    assert ws > 0, f"ws = {ws}, i = {i}, len(audio) = {audio_batch.shape[0]}"
                    print(i, ws, audio_batch.shape[0])
                    features = features[:, :ws, :]
                else:
                    features = features[:, :window_size_in_features, :]
                output = torch.cat((output, features), dim=1)
            output = output[:, :result_num_features, :]
            result = torch.cat((result, output), dim=0)
        assert result.shape[-1] == self.features_dim
        return result

    def forward(self, audio: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        return self.encode(audio, sampling_rate)