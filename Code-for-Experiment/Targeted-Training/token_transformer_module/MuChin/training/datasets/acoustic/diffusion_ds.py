import torch
import random
from typing import Optional
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from ama_prof_divi.utils import logging
logger = logging.get_logger(__name__)

class DiffusionDataset(Dataset):
    def __init__(self, windows: list[dict]):
        super(DiffusionDataset, self).__init__()
        self.windows = windows

    def __getitem__(self, index):
        assert 0 <= index < len(self.windows), f"Invalid index {index}."
        window = self.windows[index]
        data = torch.load(window["file"], map_location="cpu")
        tokens = data["audio_codes"][window["window_offset"]:window["window_offset"] + window["window_length"], :, :]
        tokens = tokens.view(-1)
        clap = data["clap"].squeeze(0)
        return {
            "tokens": tokens,
            "clap": clap,
        }

    def __len__(self):
        return len(self.windows)

    @staticmethod
    def collate_fn(batch):
        tokens = [item["tokens"] for item in batch]
        claps = [item["clap"] for item in batch]
        tokens = torch.stack(tokens, dim=0)
        claps = torch.stack(claps, dim=0)
        return {
            "tokens": tokens,
            "claps": claps,
        }

    @staticmethod
    def load_datasets(data_dir: Path, data_file_pattern: str, window_size: int, cache_file: Optional[Path] = None, use_data_files_proportion: float = 1.0, training_dataset_proportion: float = 0.8, clap_dim: int = 512, vocab_size: int = 1028, tokens_chunk_len: int = 150, tokens_num_q: int = 8, local_rank: int = 0) -> (Dataset, Dataset):
        if cache_file is not None and cache_file.exists():
            if local_rank == 0:
                logger.info(f"Loading cached data from {cache_file}")
            cache = torch.load(cache_file, map_location="cpu")
            training_windows = cache["training_windows"]
            validation_windows = cache["validation_windows"]
        else:
            data_files = list(data_dir.glob(data_file_pattern))
            random.shuffle(data_files)
            data_files = data_files[:int(len(data_files) * use_data_files_proportion)]
            random.shuffle(data_files)
            max_training_file_idx = int(len(data_files) * training_dataset_proportion)
            if local_rank == 0:
                logger.info(f"Loading {len(data_files)} data files: split into {max_training_file_idx} training files and {len(data_files) - max_training_file_idx} validation files.")
            data_files = tqdm(data_files, desc="Processing data files")
            training_windows = []
            validation_windows = []
            for file_id, file in enumerate(data_files):
                data = torch.load(file, map_location="cpu")
                assert "audio_codes" in data, f"Invalid data file {file}. Missing audio_codes."
                tokens = data["audio_codes"]
                assert tokens.shape[1:] == (tokens_chunk_len, tokens_num_q), f"Invalid tokens shape {tokens.shape} in file {file}."
                assert 0 <= tokens.min() <= tokens.max() < vocab_size, f"Invalid tokens value in file {file}."
                assert "clap" in data, f"Invalid data file {file}. Missing clap."
                clap = data["clap"]
                assert clap.shape == (1, clap_dim), f"Invalid clap shape {clap.shape} in file {file}."
                for i in range(0, tokens.shape[0], window_size):
                    window_length = min(window_size, tokens.shape[0] - i)
                    if window_length == window_size:
                        window = {
                            "file": str(file),
                            "window_offset": i,
                            "window_length": window_length,
                        }
                        if file_id < max_training_file_idx:
                            training_windows.append(window)
                        else:
                            validation_windows.append(window)
            random.shuffle(training_windows)
            random.shuffle(validation_windows)
            if local_rank == 0:
                if cache_file is not None:
                    cache = {
                        "training_windows": training_windows,
                        "validation_windows": validation_windows
                    }
                    torch.save(cache, cache_file)
                    logger.info(f"Cached data to {cache_file}.")
                logger.info(f"Training dataset length: {len(training_windows)}")
            logger.info(f"Validation dataset length: {len(validation_windows)}")
        training_dataset = DiffusionDataset(windows=training_windows)
        validation_dataset = DiffusionDataset(windows=validation_windows)
        return training_dataset, validation_dataset