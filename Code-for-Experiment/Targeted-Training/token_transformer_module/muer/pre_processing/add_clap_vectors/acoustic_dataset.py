from pathlib import Path
from re import Pattern
from tqdm import tqdm
from torch.utils.data import Dataset
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger
_logger = get_logger(__name__)

class AcousticDataset(Dataset):
    def __init__(self, index_dict: dict[str, dict[str, Path]], local_rank: int):
        super(AcousticDataset, self).__init__()
        self.local_rank = local_rank
        self.index_dict = index_dict
        self.song_id_list = list(index_dict.keys())

    @staticmethod
    def get_dataset(data_dir: Path, data_file_pattern: str, ref_dir: Path, ref_file_pattern: str, file_id_regex: Pattern, local_rank: int) -> Dataset:
        assert data_dir.exists(), f"Data directory {data_dir} does not exist."
        assert data_dir.is_dir(), f"Data directory {data_dir} is not a directory."
        assert ref_dir.exists(), f"Reference directory {ref_dir} does not exist."
        assert ref_dir.is_dir(), f"Reference directory {ref_dir} is not a directory."
        if local_rank == 0:
            _logger.info(f"Initializing acoustic dataset. Data directory: {data_dir}, reference directory: {ref_dir}...")
            _logger.info("Please wait while indexing the dataset...")
        index_dict = dict[str, dict[str, Path]]()
        data_files = data_dir.glob(data_file_pattern)
        ref_files = ref_dir.glob(ref_file_pattern)
        if local_rank == 0:
            data_files = tqdm(data_files, desc="Indexing data files")
            ref_files = tqdm(ref_files, desc="Indexing reference files")
        for file in data_files:
            file_id = AcousticDataset.get_file_id(file, file_id_regex=file_id_regex)
            index_dict[file_id] = {"data": file}
        for file in ref_files:
            file_id = AcousticDataset.get_file_id(file, file_id_regex=file_id_regex)
            if file_id in index_dict:
                index_dict[file_id]["ref"] = file
        for file_id in index_dict:
            if "ref" not in index_dict[file_id]:
                raise ValueError(f"Reference file for {file_id} is not found.")
        return AcousticDataset(index_dict, local_rank)

    @staticmethod
    def get_file_id(file_path: Path, *, file_id_regex: Pattern) -> str:
        stem = file_path.stem
        m = file_id_regex.match(stem)
        if m is not None:
            return m.group(0)
        else:
            raise ValueError(f"Bad file name format: {stem}")

    def __len__(self) -> int:
        return len(self.song_id_list)

    def __getitem__(self, idx: int) -> dict:
        song_id = self.song_id_list[idx]
        data_file = self.index_dict[song_id]["data"]
        ref_file = self.index_dict[song_id]["ref"]
        return {"song_id": song_id, "data_file": data_file, "ref_file": ref_file}

    @staticmethod
    def collate_fn(batch: list):
        assert len(batch) == 1
        return batch[0]