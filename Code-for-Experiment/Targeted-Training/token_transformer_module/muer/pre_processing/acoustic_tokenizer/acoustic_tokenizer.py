import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from pathlib import Path
from einops import rearrange
from ama_prof_divi.models.common import get_audio_utils
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.logging import get_logger
import get_encodec_wrapper
logger = get_logger(__name__)
class AcousticTokenizer:
    def __init__(self, configs: dict, device: str or torch.device, world_size: int, rank: int, local_rank: int, is_master: bool, parallel_enabled: bool = False):
        self.device = device
        self.config = configs
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_master = is_master
        self.parallel_enabled = parallel_enabled
        if self.local_rank == 0:
            logger.info(f"Initializing acoustic tokenizer. World size: {world_size}, device: '{device}', parallelism {'enabled' if self.parallel_enabled else 'disabled'}...")
        self.target_loudness = self.config["target_loudness"]
        self.window_size = self.config["window_size"]
        self.target_file_postfix = self.config["target"]["target_file_postfix"]
        self.exclude_vocals = self.config["exclude_vocals"]
        self.target_dir = Path(self.config["target"]["target_dir"])
        assert self.target_dir.exists(), f"Target directory '{self.target_dir}' does not exist."
        assert self.target_dir.is_dir(), f"Target directory '{self.target_dir}' is not a directory."
        self.states_dir = Path(self.config["target"]["states_dir"])
        assert self.states_dir.exists(), f"States directory '{self.states_dir}' does not exist."
        assert self.states_dir.is_dir(), f"States directory '{self.states_dir}' is not a directory."
        self.au_utils = get_audio_utils().to(self.device)
        self.encodec = get_encodec_wrapper().to(self.device)
        if local_rank == 0:
            logger.info("num_channels = %d", self.encodec.num_channels)
            logger.info("sampling_rate = %d", self.encodec.sampling_rate)
            logger.info("num_quantizers = %d", self.encodec.num_quantizers)
            logger.info("segment_length = %d", self.encodec.segment_length)
            logger.info("segment_stride = %d", self.encodec.segment_stride)
            logger.info("bandwidth = %d", self.encodec.bandwidth)
            logger.info("frame_rate = %d", self.encodec.frame_rate)
        self.dataset = SourceDataset(configs, local_rank)
        if self.parallel_enabled:
            self.data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=SourceDataset.collate_fn, sampler=DistributedSampler(self.dataset, num_replicas=self.world_size, rank=self.rank))
        else:
            self.data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=SourceDataset.collate_fn)
        if local_rank == 0:
            logger.info(f"Data loader size: {len(self.data_loader)}")
            logger.info(f"Acoustic tokenizer initialized.")
    def process(self):
        if self.local_rank == 0:
            logger.info(f"Start processing.")
            pbar = tqdm(total=len(self.data_loader), desc=f"Encoding")
        else:
            pbar = None
        for batch in self.data_loader:
            if "error" not in batch:
                self._process_data_batch(batch)
            if pbar is not None:
                pbar.update(1)
        if self.local_rank == 0:
            pbar.close()
    def _process_data_batch(self, batch: dict):
        song_id = batch["song_id"]
        file_group = batch["file_group"]
        audio = batch["audio_data"]
        sampling_rate = batch["sampling_rate"]
        states_path = self.states_dir / str(file_group)
        if states_path.exists():
            state_file = states_path / f"{song_id}_acoustic.done"
            if state_file.exists():
                return
        if self.exclude_vocals:
            audio_stems = self.au_utils.demucs_separate(audio, sampling_rate, stems=["drums", "bass", "other"])
            sampling_rate = self.au_utils.demucs_sampling_rate
            assert audio_stems.shape == (1, 3, self.au_utils.demucs_num_channels, audio_stems.shape[-1])
            audio = audio_stems.squeeze(0).mean(dim=0)
        if audio.shape[1] != self.encodec.num_channels or sampling_rate != self.encodec.sampling_rate:
            audio = self.au_utils.resample(audio, sampling_rate, self.encodec.sampling_rate, self.encodec.num_channels)
        sampling_rate = self.encodec.sampling_rate
        audio = self.au_utils.normalize_loudness(audio, sampling_rate, self.target_loudness)
        window_size = self.encodec.segment_stride * self.window_size
        audio_length = audio.shape[1]
        audio_codes_list = []
        audio_scale_list = []
        for offset in range(0, audio_length, window_size):
            audio_window = audio[:, offset:offset + window_size]
            encoded = self.encodec.encode(audio_window)
            encoded_window_length = audio_window.shape[1] // self.encodec.segment_stride
            audio_codes = encoded["audio_codes"][:encoded_window_length, :, :, :]
            audio_codes = rearrange(audio_codes, 's b q l -> (s b) l q').cpu()
            audio_scales = encoded["audio_scales"][:encoded_window_length, :].cpu()
            audio_codes_list.append(audio_codes)
            audio_scale_list.append(audio_scales)
        audio_codes = torch.cat(audio_codes_list, dim=0)
        assert audio_codes.shape[-1] == self.encodec.num_quantizers
        assert audio_codes.shape[1] == self.encodec.frame_rate
        audio_scales = torch.cat(audio_scale_list, dim=0)
        audio_scales = rearrange(audio_scales, 's b -> (s b)')
        audio_length_rounded_down = audio_length // sampling_rate
        audio_codes = audio_codes[:audio_length_rounded_down, :, :]
        audio_scales = audio_scales[:audio_length_rounded_down]
        result = {
            "song_id": song_id,
            "model": self.encodec.model_name,
            "audio_codes": audio_codes,
            "audio_scales": audio_scales,
            "audio_length": audio_length / sampling_rate,
            "sampling_rate": sampling_rate,
            "num_channels": self.encodec.num_channels,
            "normalized_loudness": self.target_loudness
        }
        self._save_encoded_audio(song_id, file_group, result)
    def _save_encoded_audio(self, song_id: str, file_group: int, encoded: dict):
        output_path = self.target_dir / str(file_group)
        if not output_path.exists():
            output_path.mkdir(exist_ok=True)
        output_file = output_path / (f"{song_id}" + self.target_file_postfix)
        torch.save(encoded, output_file)
        states_path = self.states_dir / str(file_group)
        if not states_path.exists():
            states_path.mkdir(exist_ok=True)
        state_file = states_path / f"{song_id}_acoustic.done"
        if not state_file.exists():
            open(state_file, "w").close()