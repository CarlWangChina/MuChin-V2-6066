import shutil
import torch
import torchaudio
import numpy as np
import pyloudnorm as pyln
from einops import rearrange
from pathlib import Path
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger

logger = get_logger(__name__)

def _normalize_loudness(audio: torch.Tensor, sampling_rate: int, target_loudness: float) -> torch.Tensor:
    meter = pyln.Meter(sampling_rate)
    audio_npd = rearrange(audio, "c n -> n c").numpy()
    loudness = meter.integrated_loudness(audio_npd)
    audio_normalized = pyln.normalize.loudness(audio_npd, loudness, target_loudness)
    return rearrange(torch.from_numpy(audio_normalized), "n c -> c n")

def _convert_mert_features(input_file: Path, output_file: Path, *, song_id: str, model_name: str, audio_length: float, feature_rate: int, feature_dim: int, mert_layer: int, normalized_loudness: float):
    assert input_file.exists(), f"Input file {str(input_file)} does not exist."
    with open(input_file, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.float32)
    data = torch.tensor(data).view(-1, feature_dim)
    mert = {
        "song_id": song_id,
        "model": model_name,
        "data": data,
        "audio_length": audio_length,
        "feature_rate": feature_rate,
        "feature_dim": feature_dim,
        "mert_layer": mert_layer,
        "normalized_loudness": normalized_loudness
    }
    torch.save(mert, output_file)

class MertEncoder:
    def __init__(self, configs: dict, feature_extractor: any, model: any):
        self.configs = configs
        self.feature_extractor = feature_extractor
        self.model = model
        self.feature_rate = configs["feature_rate"]
        self.window_size = configs["window_size"]
        self.output_layer = configs["mert_output_layer"]
        self.feature_dim = configs["feature_dim"]
        self.device = model.device
        self.output_path = Path(configs["output_path"])
        assert self.output_path.exists(), f"Output path {str(self.output_path)} does not exist."
        assert self.output_path.is_dir(), f"Output path {str(self.output_path)} is not a directory."
        self.states_path = Path(configs["states_path"])
        assert self.states_path.exists(), f"States path {str(self.states_path)} does not exist."
        assert self.states_path.is_dir(), f"States path {str(self.states_path)} is not a directory."
        if configs["duplicate_src_files"]:
            self.src_dup_path = Path(configs["src_dup_path"])
            assert self.src_dup_path.exists(), f"Duplicate source output path {str(self.src_dup_path)} does not exist."
            assert self.src_dup_path.is_dir(), f"Duplicate source output path {str(self.src_dup_path)} is not a directory."
        else:
            self.src_dup_path = None
        self.padding = self.feature_extractor.sampling_rate // self.feature_rate

    def _duplicate_source_file(self, batch: dict):
        states_path = self.states_path / str(batch["file_group"])
        if not states_path.exists():
            states_path.mkdir(exist_ok=True)
        state_file = states_path / str(batch["song_id"] + "_dup.done")
        if state_file.exists():
            return
        output_path = self.src_dup_path / str(batch["file_group"])
        if not output_path.exists():
            output_path.mkdir(exist_ok=True)
        output_file = output_path / str(batch["song_id"] + self.configs["source_file_postfix"])
        shutil.copy(batch["file_path"], output_file)
        open(state_file, "w").close()

    @torch.inference_mode()
    def _calculate_mert_features(self, audio: torch.Tensor, output_fp):
        assert audio is not None and audio.dim() == 1
        window_size_in_samples = self.window_size * self.feature_extractor.sampling_rate
        window_size_in_features = self.window_size * self.feature_rate
        for i in range(0, len(audio), window_size_in_samples):
            audio_window = audio[i: i + window_size_in_samples + self.padding]
            if audio_window.shape[0] < window_size_in_samples + self.padding:
                audio_window = torch.cat((audio_window, torch.zeros(window_size_in_samples + self.padding - audio_window.shape[0])))
            inputs = self.feature_extractor(audio_window, sampling_rate=self.feature_extractor.sampling_rate, padding=True, return_attention_mask=True, return_tensors="pt")
            inputs["input_values"] = inputs["input_values"].to(self.device)
            inputs["attention_mask"] = inputs["attention_mask"].to(self.device)
            hidden_states = self.model(**inputs, output_hidden_states=True)["hidden_states"]
            assert len(hidden_states) > self.output_layer >= 0, f"Output layer {self.output_layer} is out of range [0, {len(hidden_states) - 1}]."
            features = hidden_states[self.output_layer].to("cpu").squeeze(0)
            assert features.dim() == 2 and features.shape[-1] == self.feature_dim
            assert features.shape[0] >= window_size_in_features
            if i + window_size_in_samples >= len(audio):
                ws = (len(audio) - i) * self.feature_rate // self.feature_extractor.sampling_rate
                features = features[:ws, :]
            else:
                features = features[:window_size_in_features, :]
            features = features.type(torch.float32).flatten()
            output_fp.write(features.numpy().tobytes())

    def process_data_batch(self, batch: dict):
        if self.src_dup_path is not None:
            self._duplicate_source_file(batch)
        states_path = self.states_path / str(batch["file_group"])
        if not states_path.exists():
            states_path.mkdir(exist_ok=True)
        state_file = states_path / str(batch["song_id"] + "_mert.done")
        if state_file.exists():
            return
        try:
            audio = batch["audio_data"]
            sampling_rate = batch["sampling_rate"]
            audio = _normalize_loudness(audio, sampling_rate, self.configs["normalized_loudness"])
            mert_sampling_rate = self.feature_extractor.sampling_rate
            if sampling_rate != mert_sampling_rate:
                resampler = torchaudio.transforms.Resample(sampling_rate, mert_sampling_rate)
                audio = resampler(audio)
            assert audio.dim() == 2
            audio = torch.mean(audio, dim=0, keepdim=False)
            assert audio.dim() == 1
            output_path = self.output_path / str(batch["file_group"])
            if not output_path.exists():
                output_path.mkdir(exist_ok=True)
            tmp_output_file = output_path / str(batch["song_id"] + "_tmp.mert")
            with open(tmp_output_file, "wb") as f:
                self._calculate_mert_features(audio, f)
            output_file = output_path / str(batch["song_id"] + self.configs["target_file_postfix"])
            _convert_mert_features(tmp_output_file, output_file, song_id=batch["song_id"], model_name=self.configs["pretrained_model"], audio_length=len(audio) / self.feature_extractor.sampling_rate, feature_rate=self.feature_rate, feature_dim=self.feature_dim, mert_layer=self.output_layer, normalized_loudness=self.configs["normalized_loudness"])
            tmp_output_file.unlink()
            with open(state_file, "w") as f:
                f.close()
        except RuntimeError as e:
            logger.error(f"Failed to process file {batch['file_path']}: {str(e)}")