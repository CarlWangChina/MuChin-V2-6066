import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import logging
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.training.datasets.file_list_ds import FileListDataset
import get_melody_tokenizer

logger = logging.getLogger(__name__)

class RVQPredictor:
    def __init__(self, configs: dict, device: str or torch.device, world_size: int, rank: int, local_rank: int, is_master: bool, parallel_enabled: bool = False):
        self.device = device
        self.config = configs
        self.rank = rank
        self.local_rank = local_rank
        self.is_master = is_master
        if local_rank == 0:
            logger.info(f"Initializing RVQ predictor.")
        self.tokenizer = get_melody_tokenizer().to(self.device)
        self.data_path = Path(configs["data_path"])
        assert self.data_path.exists(), f"Data path {self.data_path} does not exist."
        assert self.data_path.is_dir(), f"Data path {self.data_path} is not a directory."
        self.dataset = FileListDataset(list(self.data_path.glob(configs["file_pattern"])))
        if parallel_enabled:
            sampler = DistributedSampler(self.dataset, num_replicas=world_size, rank=rank)
            self.data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=configs["data_loader_num_workers"], collate_fn=FileListDataset.collate_fn, sampler=sampler)
        else:
            self.data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=configs["data_loader_num_workers"], collate_fn=FileListDataset.collate_fn)
        if local_rank == 0:
            logger.info(f"Data loader size: {len(self.data_loader)}")
            logger.info(f"RVQ predictor initialized. {len(self.dataset)} files to process.")

    def process(self):
        if self.local_rank == 0:
            logger.info(f"Start processing.")
            pbar = tqdm(self.data_loader)
        else:
            pbar = None
        for batch in self.data_loader:
            file_name = batch["file_name"]
            mert = torch.load(file_name, map_location=self.device)
            if "tokens" in mert:
                if self.local_rank == 0:
                    pbar.update(1)
                continue
            mert["tokens"] = self.tokenizer.tokenize(mert["data"].to(self.device))
            mert["codebook_size"] = self.tokenizer.codebook_size
            mert["vocab_size"] = self.tokenizer.vocab_size
            torch.save(mert, file_name)
            if self.local_rank == 0:
                pbar.update(1)