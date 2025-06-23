import json
from pathlib import Path
import torchaudio
from torch.utils.data import Dataset
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger

logger = get_logger(__name__)

class Mp3SourceDataset(Dataset):
    def __init__(self, control_files: [Path], source_file_postfix: str, local_rank: int = 0, output_group_size=10000, use_abs_path: bool = True, base_data_dir: Path = None):
        super(Mp3SourceDataset, self).__init__()
        self.local_rank = local_rank
        self.output_group_size = output_group_size
        self.source_file_postfix = source_file_postfix
        assert use_abs_path or base_data_dir is not None, "Either use absolute path or provide base data directory."
        for control_file in control_files:
            assert control_file.exists(), f"Data control file {control_file} does not exist."
            assert control_file.is_file(), f"Data control file {control_file} is not a file."
        song_id_paths = {}
        for control_file in control_files:
            with open(control_file, "r", encoding="utf-8") as f:
                for line in f:
                    json_data = json.loads(line)
                    if "path" in json_data:
                        p = Path(json_data["path"] + source_file_postfix)
                        song_id = p.stem + p.suffix
                        assert song_id.endswith(self.source_file_postfix)
                        song_id = song_id[:song_id.find(self.source_file_postfix)]
                        if song_id in song_id_paths and local_rank == 0:
                            logger.warning(f"Dataset: Song id {song_id} is duplicated in control file {control_file}.")
                        if use_abs_path:
                            song_id_paths[song_id] = p
                        else:
                            song_id_paths[song_id] = base_data_dir / p
        self.song_paths = list(song_id_paths.values())

    def __len__(self) -> int:
        return len(self.song_paths)

    def __getitem__(self, idx):
        file_path = self.song_paths[idx]
        assert file_path.exists(), f"Data path {file_path} does not exist."
        assert file_path.is_file(), f"Data path {file_path} is not a file."
        error = None
        try:
            audio, sampling_rate = torchaudio.load(file_path)
        except RuntimeError as e:
            error = e
            logger.error(f"Failed to load file {file_path}: {str(e)}")
            audio = None
            sampling_rate = None
        song_id = file_path.stem + file_path.suffix
        assert song_id.endswith(self.source_file_postfix)
        song_id = song_id[:song_id.find(self.source_file_postfix)]
        if error is None:
            return {
                "song_id": song_id,
                "file_path": file_path,
                "file_group": idx // self.output_group_size,
                "audio_data": audio,
                "sampling_rate": sampling_rate
            }
        else:
            return {
                "song_id": song_id,
                "file_path": file_path,
                "error": error
            }

    @staticmethod
    def collate_fn(batch: list):
        assert len(batch) == 1
        return batch[0]