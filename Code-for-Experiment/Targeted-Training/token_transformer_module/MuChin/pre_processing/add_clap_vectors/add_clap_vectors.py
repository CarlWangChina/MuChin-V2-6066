import torch
import re
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, DistributedSampler
import logging
logger = logging.getLogger(__name__)
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.audioutils import get_audio_utils
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.pre_processing.add_clap_vectors.acoustic_dataset import AcousticDataset
import get_prompt_encoder

class AddClapVectors:
    def __init__(self, configs: dict, device: str or torch.device, world_size: int, rank: int, local_rank: int, is_master: bool, parallel_enabled: bool = False):
        self.device = device
        self.config = configs
        self.rank = rank
        self.local_rank = local_rank
        self.is_master = is_master
        self.world_size = world_size
        self.parallel_enabled = parallel_enabled
        if self.local_rank == 0:
            logger.info(f"Initializing: World size: {world_size}, device: '{device}', parallelism {'enabled' if self.parallel_enabled else 'disabled'}...")
        self.target_loudness = self.config["target_loudness"]
        self.audio_utils = get_audio_utils()
        data_dir = Path(self.config["data_dir"])
        data_file_pattern = self.config["data_file_pattern"]
        reference_dir = Path(self.config["ref_dir"])
        ref_file_pattern = self.config["ref_file_pattern"]
        file_id_regex = re.compile(self.config["file_id_regex"])
        self.dataset = AcousticDataset.get_dataset(data_dir=data_dir, data_file_pattern=data_file_pattern, ref_dir=reference_dir, ref_file_pattern=ref_file_pattern, file_id_regex=file_id_regex, local_rank=self.local_rank)
        if self.parallel_enabled:
            self.data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=AcousticDataset.collate_fn, sampler=DistributedSampler(self.dataset, num_replicas=self.world_size, rank=self.rank))
        else:
            self.data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=AcousticDataset.collate_fn)
        self.prompt_encoder = get_prompt_encoder(device=self.device)
        if self.local_rank == 0:
            logger.info(f"Data loader size: {len(self.data_loader)}")
            logger.info(f"Clap vectors adder initialized.")

    def process(self):
        data_loader = self.data_loader
        if self.local_rank == 0:
            logger.info(f"Start processing.")
            data_loader = tqdm(data_loader, desc="Processing")
        for batch in data_loader:
            data_file = batch["data_file"]
            ref_file = batch["ref_file"]
            data_dict = torch.load(data_file)
            if "clap" in data_dict:
                continue
            data_dict["clap"] = self._get_clap_encoding(ref_file)
            torch.save(data_dict, data_file)
        if self.local_rank == 0:
            logger.info("Done.")

    def _get_clap_encoding(self, audio_file: Path) -> torch.Tensor:
        audio, sampling_rate = self.audio_utils.load_audio(audio_file)
        audio = audio.mean(dim=0, keepdim=True)
        audio = self.audio_utils.resample(audio, sampling_rate, self.prompt_encoder.sampling_rate)
        audio = self.audio_utils.normalize_loudness(audio, sampling_rate, self.target_loudness).cpu()
        return self.prompt_encoder.get_audio_embedding(audio)