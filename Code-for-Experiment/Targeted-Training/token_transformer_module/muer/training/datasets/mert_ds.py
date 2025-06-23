import random
import torch
from typing import Optional
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from ama_prof_divi.utils import logging
logger = logging.get_logger(__name__)
MERT_INDEX_FILE = "mert-index.pt"

@dataclass
class MertWindowInfo:
    file_path: str
    window_start: int
    window_length: int

class MertDataset(Dataset):
    def __init__(self, 
                 mert_windows: [MertWindowInfo],
                 window_size: int,
                 feature_dim: int,
                 feature_rate: int,
                 content: str,
                 last_window: str,
                 local_rank: int = 0):
        super(MertDataset, self).__init__()
        assert last_window in ["drop", "cutoff", "padding"], \
            f"The last window strategy should be one of 'drop', 'cutoff' or 'padding', but got {last_window}."
        assert content in ["vectors", "tokens"], \
            f"The content should be one of 'vectors' or 'tokens', but got {content}."
        self.feature_dim = feature_dim
        self.feature_rate = feature_rate
        self.window_size = window_size * self.feature_rate
        self.last_window = last_window
        self.content = content
        self.local_rank = local_rank
        self.mert_windows = mert_windows

    def __len__(self):
        return len(self.mert_windows)

    def __getitem__(self, index):
        assert 0 <= index < len(self), f"Index {index} out of range [0, {len(self)})."
        mert_window = self.mert_windows[index]
        mert = torch.load(mert_window.file_path, map_location="cpu")
        if self.content == "vectors":
            mert_data = mert['data'][mert_window.window_start:mert_window.window_start + mert_window.window_length, :]
        else:
            assert self.content == "tokens", f"Unknown content type {self.content}."
            mert_data = mert['tokens'][mert_window.window_start:mert_window.window_start + mert_window.window_length]
        del mert
        if self.last_window == "padding" and mert_data.shape[0] < self.window_size:
            if self.content == "vectors":
                mert_data = torch.cat([mert_data, torch.zeros(self.window_size - mert_data.shape[0], self.feature_dim)], dim=0)
            else:
                assert self.content == "tokens", f"Unknown content type {self.content}."
                mert_data = torch.cat([mert_data, torch.zeros((self.window_size - mert_data.shape[0],), dtype=torch.long)], dim=0)
        return mert_data

    @staticmethod
    def collate_fn(batch):
        return torch.stack(batch, dim=0)

    @staticmethod
    def load_datasets(data_dir: Path,
                      mert_file_pattern: str,
                      cache_file: Optional[Path] = None,
                      index_file: Optional[Path] = None,
                      window_size: int = 20,
                      feature_dim: int = 1024,
                      feature_rate: int = 75,
                      stride: int = -1,
                      content: str = "vectors",
                      last_window: str = "drop",
                      use_data_files_proportion: float = 1.0,
                      training_dataset_proportion: float = 0.8,
                      local_rank: int = 0) -> (Dataset, Dataset):
        assert data_dir.exists(), f"Data directory {data_dir} does not exist."
        assert data_dir.is_dir(), f"Data directory {data_dir} is not a directory."
        assert last_window in ["drop", "cutoff", "padding"], \
            f"The last window strategy should be one of 'drop', 'cutoff' or 'padding', but got {last_window}."
        assert content in ["vectors", "tokens"], \
            f"The content should be one of 'vectors' or 'tokens', but got {content}."
        if cache_file is not None and cache_file.exists():
            if local_rank == 0:
                logger.info("Loading cached Mert dataset from %s.", cache_file)
            cache = torch.load(cache_file, map_location="cpu")
            training_mert_windows = cache["training_mert_windows"]
            validation_mert_windows = cache["validation_mert_windows"]
        else:
            window_size *= feature_rate
            if stride <= 0:
                stride = window_size
            mert_windows: [MertWindowInfo] = []
            if index_file is None:
                index_file = data_dir / MERT_INDEX_FILE
            if index_file.exists():
                if local_rank == 0:
                    logger.info("Loading Mert from index file %s.", index_file)
                mert_index = torch.load(index_file, map_location="cpu")
                assert feature_dim == mert_index["feature_dim"], \
                    f"Feature dimension mismatch: {feature_dim} vs {mert_index['feature_dim']}."
                assert feature_rate == mert_index["feature_rate"], \
                    f"Feature rate mismatch: {feature_rate} vs {mert_index['feature_rate']}."
                mert_list = mert_index["mert_files"]
                random.shuffle(mert_list)
                mert_list = mert_list[:int(len(mert_list) * use_data_files_proportion)]
                for mert_entry in mert_list:
                    file = data_dir / str(mert_entry["file_group"]) / mert_entry["file_name"]
                    mert_len = mert_entry["data_length"]
                    for i in range(0, mert_len, stride):
                        window_start = i
                        window_length = min(window_size, mert_len - i)
                        if window_length < window_size:
                            if last_window == "drop":
                                break
                        mert_windows.append(MertWindowInfo(str(file), window_start, window_length))
            else:
                logger.warning("Index file is not found.  Walking through all Mert files could be very slow."
                               "  It is recommended to create an index file for the Mert data.  Run the script"
                               " 'scripts/extract_mert_features/make_mert_indices' to create the index file.")
                mert_files = list(data_dir.glob(mert_file_pattern))
                random.shuffle(mert_files)
                mert_files = mert_files[:int(len(mert_files) * use_data_files_proportion)]
                if local_rank == 0:
                    if len(mert_files) > 0:
                        logger.info("Found %d Mert files.", len(mert_files))
                    else:
                        logger.error("No Mert files found in %s.", data_dir)
                if local_rank == 0:
                    pbar = tqdm(mert_files, desc="Processing Mert files")
                else:
                    pbar = None
                for file in mert_files:
                    mert = torch.load(file, map_location="cpu")
                    if feature_dim != mert['feature_dim'] or feature_rate != mert['feature_rate']:
                        logger.error(f"Feature rate or dimension mismatch for {file}. "
                                     f"Expected {feature_rate} by {feature_dim}, "
                                     f"actual {mert['feature_rate']} by {mert['feature_dim']}." )
                        if pbar is not None:
                            pbar.update(1)
                        continue
                    mert_len = mert['data'].shape[0]
                    for i in range(0, mert_len, stride):
                        window_start = i
                        window_length = min(window_size, mert_len - i)
                        if window_length < window_size:
                            if last_window == "drop":
                                break
                        mert_windows.append(MertWindowInfo(str(file), window_start, window_length))
                    del mert
                    if pbar is not None:
                        pbar.update(1)
                if local_rank == 0:
                    pbar.close()
            if local_rank == 0:
                logger.info("Found %d Mert windows.", len(mert_windows))
            random.shuffle(mert_windows)
            dataset_len = len(mert_windows)
            training_dataset_len = int(dataset_len * training_dataset_proportion)
            training_mert_windows = mert_windows[:training_dataset_len]
            validation_mert_windows = mert_windows[training_dataset_len:dataset_len]
            if local_rank == 0 and cache_file is not None:
                cache = {
                    "training_mert_windows": training_mert_windows,
                    "validation_mert_windows": validation_mert_windows
                }
                torch.save(cache, cache_file)
                logger.info("Cached Mert dataset to %s.", cache_file)
        if local_rank == 0:
            logger.info("Training Mert dataset: %d windows.", len(training_mert_windows))
            logger.info("Validation Mert dataset: %d windows.", len(validation_mert_windows))
        train_dataset = MertDataset(mert_windows=training_mert_windows,
                                    window_size=window_size,
                                    feature_dim=feature_dim,
                                    feature_rate=feature_rate,
                                    content=content,
                                    last_window=last_window,
                                    local_rank=local_rank)
        valid_dataset = MertDataset(mert_windows=validation_mert_windows,
                                    window_size=window_size,
                                    feature_dim=feature_dim,
                                    feature_rate=feature_rate,
                                    content=content,
                                    last_window=last_window,
                                    local_rank=local_rank)
        return train_dataset, valid_dataset