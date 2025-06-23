import torchaudio
import json
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger
logger = get_logger(__name__)

class SourceDataset(Dataset):
    def __init__(self, configs: dict, local_rank: int):
        super(SourceDataset, self).__init__()
        self.local_rank = local_rank
        self.ds_configs = configs["data_source"]
        self.output_file_group_size = configs["target"]["group_size"]
        use_absolute_path = self.ds_configs["use_absolute_path"]
        source_dir = self.ds_configs["source_dir"]

        if not use_absolute_path:
            assert source_dir is not None, "Source directory is not provided."
            assert Path(source_dir).exists(), f"Source directory {source_dir} does not exist."
            assert Path(source_dir).is_dir(), f"Source directory {source_dir} is not a directory."
            source_dir = Path(source_dir).absolute()
        else:
            source_dir = None
        control_file_paths = self.ds_configs["data_control_files"]
        source_file_postfix = self.ds_configs["source_file_postfix"]
        current_path = Path(configs["current_path"])

        if self.local_rank == 0:
            logger.info(f"Initializing source dataset by walking through all audio files. ")
        self.song_id_paths = {}

        for control_file in control_file_paths:
            try:
                with open(current_path / control_file, "r", encoding="utf-8") as f:
                    for line in f:
                        json_data = json.loads(line)
                        if "path" in json_data:
                            p = [Path(json_data["path"] + postfix) for postfix in source_file_postfix]
                            assert len(p) > 0, "Source file postfix is empty."
                            song_id = p[0].stem + p[0].suffix
                            assert song_id.endswith(source_file_postfix[0])
                            song_id = song_id[:song_id.find(source_file_postfix[0])]
                            if song_id in self.song_id_paths and local_rank == 0:
                                logger.warning(f"Dataset: Song id {song_id} is duplicated in control file {control_file}.")
                            song_file_path = self._get_song_file(p, use_absolute_path, source_dir)
                            if song_file_path is not None:
                                self.song_id_paths[song_id] = song_file_path
                            else:
                                logger.warning(f"Dataset: Song file {song_id} does not exist.")
            except RuntimeError as e:
                logger.error(f"Error opening control file {str(control_file)}.")
                raise e
        if local_rank == 0:
            logger.info(f"Source dataset initialized. Total {len(self.song_id_paths)} songs.")
        self.song_id_list = list(self.song_id_paths.keys())

    def __len__(self) -> int:
        return len(self.song_id_paths)

    def __getitem__(self, idx):
        song_id = self.song_id_list[idx]
        file_path = Path(self.song_id_paths[song_id])
        try:
            audio, sampling_rate = torchaudio.load(file_path)
            return {
                "song_id": song_id,
                "file_path": file_path,
                "file_group": idx // self.output_file_group_size,
                "audio_data": audio,
                "sampling_rate": sampling_rate
            }
        except RuntimeError as e:
            error = e
            logger.error(f"Failed to load file {file_path}: {str(e)}")
            return {
                "song_id": song_id,
                "file_path": file_path,
                "error": error
            }

    @staticmethod
    def _get_song_file(paths: [Path], use_absolute_path: bool, base_data_dir: Optional[Path]) -> Optional[str]:
        for p in paths:
            if not use_absolute_path:
                p = base_data_dir / p
            if p.exists() and p.is_file():
                return str(p.absolute())
        return None

    @staticmethod
    def collate_fn(batch: list):
        assert len(batch) == 1
        return batch[0]