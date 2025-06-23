from pathlib import Path
from typing import List
from torch.utils.data import Dataset

class FileListDataset(Dataset):
    def __init__(self, file_list: List[Path]):
        super(FileListDataset, self).__init__()
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        return {
            "file_name": self.file_list[idx]
        }

    @staticmethod
    def collate_fn(batch: list):
        assert len(batch) == 1
        return batch[0]