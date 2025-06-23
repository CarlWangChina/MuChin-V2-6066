import os
import sys
import logging
import json
import shutil
import pyloudnorm as pyln
from einops import rearrange
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
import torch
import torchaudio
from Code_for_Experiment.Metrics.music_understanding_model.apex.apex.parallel import distributed as dist
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import Wav2Vec2FeatureExtractor, AutoModel

if sys.argv[1] == "330":
    CONFIG_FILE = "configs/mert330.yaml"
elif sys.argv[1] == "95":
    CONFIG_FILE = "configs/mert95.yaml"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _normalize_loudness(audio: torch.Tensor, sampling_rate: int, target_loudness: float) -> torch.Tensor:
    meter = pyln.Meter(sampling_rate)
    audio_npd = rearrange(audio, "c n -> n c").numpy()
    loudness = meter.integrated_loudness(audio_npd)
    audio_normalized = pyln.normalize.loudness(audio_npd, loudness, target_loudness)
    return rearrange(torch.from_numpy(audio_normalized), "n c -> c n")

class MertProcessingDataset(Dataset):
    def __init__(self, control_files: [Path], source_file_postfix: str, rank: int = 0, output_group_size: int = 10000, use_abs_path: bool = True, base_data_dir: Path = None):
        super(MertProcessingDataset, self).__init__()
        self.rank = rank
        self.output_group_size = output_group_size
        self.source_file_postfix = source_file_postfix
        assert use_abs_path or base_data_dir is not None, "Either use absolute path or provide base data directory."
        for control_file in control_files:
            assert control_file.exists(), f"Data control file {control_file} does not exist."
            assert control_file.is_file(), f"Data control file {control_file} is not a file."
        song_id_paths = {}
        for control_file in control_files:
            with open(control_file, "r", encoding="utf-8") as f:
                for line in f:
                    json_data = json.loads(line)
                    if "path" in json_data:
                        p = Path(json_data["path"] + source_file_postfix)
                        song_id = p.stem
                        if song_id in song_id_paths and rank == 0:
                            logger.warning(f"Dataset: Song id {song_id} is duplicated in control file {control_file}.")
                        if use_abs_path:
                            song_id_paths[song_id] = p
                        else:
                            song_id_paths[song_id] = base_data_dir / p
        self.song_paths = list(song_id_paths.values())

    def __len__(self) -> int:
        return len(self.song_paths)

    def __getitem__(self, idx):
        file_path = self.song_paths[idx]
        assert file_path.exists(), f"Data path {file_path} does not exist."
        assert file_path.is_file(), f"Data path {file_path} is not a file."
        error = None
        try:
            audio, sampling_rate = torchaudio.load(file_path)
        except RuntimeError as e:
            error = e
            logger.error(f"Failed to load file {file_path}: {str(e)}")
            audio = None
            sampling_rate = None
        song_id = file_path.stem + file_path.suffix
        assert song_id.endswith(self.source_file_postfix)
        song_id = song_id[:song_id.find(self.source_file_postfix)]
        if error is None:
            return {
                "song_id": song_id,
                "file_path": file_path,
                "file_group": idx // self.output_group_size,
                "audio_data": audio,
                "sampling_rate": sampling_rate
            }
        else:
            return {
                "song_id": song_id,
                "file_path": file_path,
                "error": error
            }

    @staticmethod
    def collate_fn(batch: list):
        assert len(batch) == 1
        return batch[0]

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
            states_path.mkdir()
        state_file = states_path / str(batch["song_id"] + "_dup.done")
        if state_file.exists():
            return
        output_path = self.src_dup_path / str(batch["file_group"])
        if not output_path.exists():
            output_path.mkdir()
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
            states_path.mkdir()
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
                output_path.mkdir()
            output_file = output_path / str(batch["song_id"] + self.configs["target_file_postfix"])
            with open(output_file, "wb") as f:
                self._calculate_mert_features(audio, f)
            open(state_file, "w").close()
        except RuntimeError as e:
            logger.error(f"Failed to process file {batch['file_path']}: {str(e)}")

def _setup(rank: int, configs: dict, world_size: int):
    os.environ['MASTER_ADDR'] = configs["master_addr"]
    os.environ['MASTER_PORT'] = str(configs["master_port"])
    dist.init_process_group(backend=configs["dist_backend"], rank=rank, world_size=world_size)

def _cleanup():
    dist.destroy_process_group()

def main(rank: int, configs: dict):
    if configs["device"] == "cuda":
        configs["device"] = torch.device(f"cuda:{rank}")
    else:
        configs["device"] = torch.device(configs["device"])
    logger.info(f"Process {rank} started.")
    logger.debug(f"Configs: {configs}")
    logger.debug(f"sys.path: {sys.path}")
    _setup(rank, configs, configs["world_size"])
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(configs["pretrained_model"], cache_dir=configs["cache_dir"], trust_remote_code=True)
    assert feature_extractor is not None
    model = AutoModel.from_pretrained(configs["pretrained_model"], cache_dir=configs["cache_dir"], trust_remote_code=True).to(configs["device"])
    assert model is not None
    data_control_files = [Path(configs["root_path"]) / f for f in configs["data_control_files"]]
    dataset = MertProcessingDataset(control_files=data_control_files, source_file_postfix=configs["source_file_postfix"], rank=rank, use_abs_path=True)
    logger.info(f"Process {rank} Dataset created with {len(dataset)} songs.")
    data_sampler = DistributedSampler(dataset, num_replicas=configs["world_size"], rank=rank)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=configs["data_loader_num_workers"], collate_fn=MertProcessingDataset.collate_fn, sampler=data_sampler)
    logger.info(f"Process {rank} Data loader size: {len(data_loader)}")
    encoder = MertEncoder(configs=configs, feature_extractor=feature_extractor, model=model)
    for batch in tqdm(data_loader, desc=f"Rank {rank}"):
        if "error" not in batch:
            encoder.process_data_batch(batch)
    _cleanup()
    logger.info(f"Process {rank} Done -- Success.\n")

if __name__ == "__main__":
    root_path = Path(__file__).parent
    logger.info("Extracting mert features ...")
    _configs = OmegaConf.load(CONFIG_FILE)
    _configs = _configs.mert_extractor
    _configs.root_path = str(root_path)
    if torch.cuda.is_available():
        _configs.world_size = torch.cuda.device_count()
        _configs.device = "cuda"
        logger.info(f"Using {_configs.world_size} CUDA GPUs.")
        torch.multiprocessing.spawn(main, args=(OmegaConf.to_container(_configs, resolve=True),), join=True, nprocs=_configs.world_size)
    elif torch.backends.mps.is_available():
        _configs.world_size = 1
        _configs.device = "mps"
        logger.info(f"Using Apple MPS.")
        main(0, OmegaConf.to_container(_configs, resolve=True))
    else:
        _configs.world_size = 1
        _configs.device = "cpu"
        logger.info(f"Using CPU.")
        main(0, OmegaConf.to_container(_configs, resolve=True))