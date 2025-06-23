from pathlib import Path
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import Wav2Vec2FeatureExtractor, AutoModel
import get_logger
from .source_dataset import Mp3SourceDataset
import MertEncoder
logger = get_logger(__name__)

class MertExtractor:
    def __init__(self, configs: dict, device: str or torch.device, world_size: int, rank: int, local_rank: int, is_master: bool, parallel_enabled: bool = False):
        self.device = device
        self.config = configs
        self.rank = rank
        self.local_rank = local_rank
        self.is_master = is_master
        feature_extractor = (Wav2Vec2FeatureExtractor.from_pretrained(configs["pretrained_model"], cache_dir=configs["cache_dir"], trust_remote_code=True))
        model = (AutoModel.from_pretrained(configs["pretrained_model"], cache_dir=configs["cache_dir"], trust_remote_code=True)).to(self.device)
        data_control_files = [Path(configs["current_path"]) / f for f in configs["data_control_files"]]
        self.dataset = Mp3SourceDataset(control_files=data_control_files, source_file_postfix=configs["source_file_postfix"], local_rank=local_rank, use_abs_path=True)
        if local_rank == 0:
            logger.info(f"Dataset created with {len(self.dataset)} songs.")
        if parallel_enabled:
            sampler = DistributedSampler(self.dataset, num_replicas=world_size, rank=rank)
            self.data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=configs["data_loader_num_workers"], collate_fn=Mp3SourceDataset.collate_fn, sampler=sampler)
        else:
            self.data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=configs["data_loader_num_workers"], collate_fn=Mp3SourceDataset.collate_fn)
        if local_rank == 0:
            logger.info(f"Data loader size: {len(self.data_loader)}")
        self.encoder = MertEncoder(configs=configs, feature_extractor=feature_extractor, model=model)
        if local_rank == 0:
            logger.info(f"Mert extractor initialized.")

    def process(self):
        if self.local_rank == 0:
            logger.info(f"Start processing.")
            pbar = tqdm(total=len(self.data_loader), desc=f"Encoding")
        else:
            pbar = None
        for batch in self.data_loader:
            if "error" not in batch:
                self.encoder.process_data_batch(batch)
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()
        if self.local_rank == 0:
            logger.info(f"Processing finished.")