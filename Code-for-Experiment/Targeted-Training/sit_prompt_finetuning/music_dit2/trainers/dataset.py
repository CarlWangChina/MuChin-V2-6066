import os
import sys
import torch
import deepspeed
import numpy as np
import random
from typing import Dict, List, Any
from pathlib import Path
from typing import Optional
from collections import OrderedDict
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.hparams import get_hparams
import Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.dist_wrapper as dists

NUM_LOCAL_RANKS = 8
DTYPE_MAP = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.int16: np.int16,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.bool: np.bool_,
}

class DiTDataset():
    def __init__(self,
                 data_folders: Dict[str, str],
                 dimensions: Dict[str, str],
                 data_types: Dict[str, str],
                 vocabulary: str,
                 rank: int,
                 start_proportion: float,
                 end_proportion: float):
        assert len(data_folders) > 0, "Data folders are empty."
        self.dimensions = dimensions
        self.data_types = data_types
        self.group_num = len([group_name for group_name, data_folder in data_folders.items() if data_folder is not None])
        self.group_path_items = OrderedDict()
        with open(vocabulary, "r") as f:
            phoneme_vocab = f.read().splitlines()
        self.vocab = {note: i for i, note in enumerate(phoneme_vocab)}
        for group_name, data_folder in data_folders.items():
            if data_folder is not None:
                for subdir, _, files in os.walk(data_folder):
                    for file in files:
                        song_id = os.path.splitext(file)[0]
                        if song_id not in self.group_path_items:
                            self.group_path_items[song_id] = OrderedDict()
                        self.group_path_items[song_id][group_name] = os.path.join(subdir, file)
        keys_to_delete = []
        for key, value in self.group_path_items.items():
            if len(value) != self.group_num:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del self.group_path_items[key]
        self.group_path_items = list(self.group_path_items.items())
        total_items = len(self.group_path_items)
        logger = deepspeed.logger
        logger.info("Total number of data: %d", len(self.group_path_items))
        items_per_rank = total_items // NUM_LOCAL_RANKS
        start_index = rank * items_per_rank + int(start_proportion * items_per_rank)
        end_index = start_index + int(items_per_rank * (end_proportion - start_proportion))
        self.rank_path_items = self.group_path_items[start_index:end_index]
        logger.info("Rank %d has %d data items.", rank, len(self.rank_path_items))

    def __len__(self):
        return len(self.rank_path_items)

    def trim_to_length_(self,
                        length: int):
        if 0 <= length < len(self.rank_path_items):
            self.rank_path_items = self.rank_path_items[:length]

    def _load(self, path):
        assert os.path.exists(path), f"File {path} does not exist."
        if path.endswith(".pt"):
            return torch.load(path)
        elif path.endswith(".npy"):
            return np.load(path)
        elif path.endswith(".txt"):
            with open(path, "r") as f:
                return f.read().replace("\n", " ")

    def _phoneme_to_id(self, phoneme, dtype):
        lyrics_labels = torch.tensor([self.vocab[note] for note in phoneme.split()], dtype=dtype)
        return lyrics_labels

    def _convert_tensor(self, data, dimensions, dtype):
        if isinstance(data, torch.Tensor):
            data = data.to(dtype)
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(DTYPE_MAP[dtype])
        elif isinstance(data, dict):
            if "clap" in list(data.keys())[0]:
                choice_key = random.randint(0, len(data.keys()) - 1)
                data = self._convert_tensor(data[list(data.keys())[choice_key]], dimensions, dtype)
            else:
                data_key = max(data.keys(), key=lambda k: sys.getsizeof(data[k]))
                data = self._convert_tensor(data[data_key], dimensions, dtype)
        else:
            raise Exception("Invalid data type")
        return data.view(*dimensions, -1)

    def __getitem__(self, index):
        assert index < len(self), f"Index out of range: {index} >= {len(self)}."
        song_id, group_dict = self.rank_path_items[index]
        result = {
            "song_id": song_id
        }
        for group_id, song_group_path in group_dict.items():
            group_data = self._load(song_group_path)
            if group_id == "lyrics":
                result[group_id] = self._phoneme_to_id(group_data, dtype=eval("torch." + self.data_types[group_id]))
            else:
                result[group_id] = self._convert_tensor(group_data, dimensions=eval(self.dimensions[group_id]), dtype=eval("torch." + self.data_types[group_id]))
        return result

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        song_ids = []
        for item in batch:
            song_ids.append(item["song_id"])
        group_names = list(batch[0].keys())
        for i in range(1, len(batch)):
            assert set(group_names) == set(batch[i].keys()), "Group names are not consistent."
        group_names.remove("song_id")
        collated_data = {
            "song_ids": song_ids
        }
        for group_name in group_names:
            data_items = [item[group_name] for item in batch]
            if len(data_items) <= 1:
                collated_data[group_name] = {
                    "data": data_items[0],
                    "padding_mask": torch.ones(data_items[0].size(), torch.bool)
                }
                continue
            max_length = max([data.size(-1) for data in data_items])
            output_shape = (len(batch),) + data_items[0].shape[:-1] + (max_length,)
            data_tensor = torch.zeros(output_shape, dtype=data_items[0].dtype)
            padding_mask = torch.zeros((len(batch), max_length), dtype=torch.bool)
            for i, data in enumerate(data_items):
                data_tensor[i, ..., :data.size(-1)] = data[..., :data.size(-1)]
                padding_mask[i, :data.size(-1)] = True
            collated_data[group_name] = {
                "data": data_tensor,
                "padding_mask": padding_mask
            }
        return collated_data

    @staticmethod
    def make_datasets():
        root_dir = Path(__file__).parent.parent.parent
        hparams = get_hparams(root_dir).training.dataset
        rank = dists.get_rank()
        logger = deepspeed.logger
        logger.info("Making training and validation datasets for rank %d...", rank)
        training_end_proportion = hparams.use_data_proportion * hparams.train_proportion
        validation_end_proportion = hparams.use_data_proportion
        train_dataset = DiTDataset(
            data_folders=hparams.data_folders,
            dimensions=hparams.dimensions,
            data_types=hparams.data_types,
            vocabulary=hparams.vocabulary,
            rank=rank,
            start_proportion=0.0,
            end_proportion=training_end_proportion)
        validation_dataset = DiTDataset(
            data_folders=hparams.data_folders,
            dimensions=hparams.dimensions,
            data_types=hparams.data_types,
            vocabulary=hparams.vocabulary,
            rank=rank,
            start_proportion=training_end_proportion,
            end_proportion=validation_end_proportion)
        return train_dataset, validation_dataset