import torch
import random
from typing import Tuple
from torch.utils.data import Dataset
from music_dit.utils import get_logger, get_hparams

logger = get_logger(__name__)

class VAETrainingDataset(Dataset):
    def __init__(self):
        hparams = get_hparams()
        self.chunk_size = hparams.vae.chunk_length
        self.sampling_rate = hparams.vae.sampling_rate
        self.num_channels = hparams.vae.num_channels
        self.data_length = 10000

    def __len__(self) -> int:
        return self.data_length

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.randn(self.num_channels, self.chunk_size)

    @staticmethod
    def collate_fn(samples):
        return torch.stack(samples, dim=0)

    @staticmethod
    def load_datasets(use_data_files_proportion: float = 1.0,
                      training_dataset_proportion: float = 0.9) -> Tuple['VAETrainingDataset', 'VAETrainingDataset']:
        return (VAETrainingDataset(),
                VAETrainingDataset())