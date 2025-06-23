import os
import csv
import json
import librosa
from torch.utils.data import Dataset
from pathlib import Path
import pyloudnorm as pyln

class AudioDataset(Dataset):
    def __init__(self, pathes: list[str], rootPath: str, outputPath: str, statePath: str, rank: int = 0, getLength: bool = False, output_group_size=10000):
        super(AudioDataset, self).__init__()
        self.rank = rank
        self.output_group_size = output_group_size
        self.getLength = getLength
        self.song_pathes = []
        for path in pathes:
            path = os.path.abspath(path)
            rootPath = os.path.abspath(rootPath)
            if os.path.commonpath([path, rootPath]) != rootPath:
                continue
            relative_path = os.path.relpath(path, rootPath)
            absolute_path = os.path.abspath(os.path.join(outputPath, relative_path))
            state_path = os.path.abspath(os.path.join(statePath, relative_path))
            if os.path.exists(path) and not os.path.exists(state_path):
                self.song_pathes.append((path, absolute_path, state_path))

    def __len__(self) -> int:
        return len(self.song_pathes)

    def __getitem__(self, idx):
        file_path_pair = self.song_pathes[idx]
        file_path = Path(file_path_pair[0])
        out_path = Path(file_path_pair[1])
        state_path = Path(file_path_pair[2])
        assert file_path.exists(), f"Data path {file_path} does not exist."
        assert file_path.is_file(), f"Data path {file_path} is not a file."
        error = None
        try:
            audio, sampling_rate = librosa.load(file_path, sr=48000)
            assert len(audio.shape) == 1
            assert sampling_rate == 48000
            meter = pyln.Meter(sampling_rate)
            loudness = meter.integrated_loudness(audio)
            audio = pyln.normalize.loudness(audio, loudness, -12.0)
        except Exception as e:
            error = e
            print(f"Failed to load file {file_path}: {str(e)}")
            audio = None
            sampling_rate = None
        if error is None:
            return {
                "file_path": file_path,
                "out_path": out_path,
                "state_path": state_path,
                "file_group": idx // self.output_group_size,
                "audio_data": audio,
                "sampling_rate": sampling_rate
            }
        else:
            return {
                "file_path": file_path,
                "out_path": out_path,
                "error": error
            }

    @staticmethod
    def collate_fn(batch: list):
        assert len(batch) == 1
        return batch[0]

class LrcDataset(Dataset):
    def __init__(self, indexPathes: str, rootPath: str, outputPath: str, statePath: str, audioPath: str, rank: int = 0, output_group_size=10000):
        super(LrcDataset, self).__init__()
        self.rank = rank
        self.output_group_size = output_group_size
        self.song_pathes = []
        with open(indexPathes, newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)
            for row in csv_reader:
                dbId = row[0]
                songId = row[1]
                songName = row[2]
                songPath = f"{rootPath}/{dbId}.txt"
                relative_path = f"{songId}.txt"
                audio_path = f"{audioPath}/{songId}_src.mp3"
                absolute_path = os.path.abspath(os.path.join(outputPath, relative_path)).replace(".txt", ".pt")
                state_path = os.path.abspath(os.path.join(statePath, relative_path))
                if os.path.exists(songPath) and os.path.exists(audio_path) and not os.path.exists(state_path):
                    self.song_pathes.append((songPath, absolute_path, state_path, dbId, songName, songId, audio_path))

    def readfile(self, file_path: str) -> str:
        concatenated_text = ""
        with open(file_path, 'r') as file:
            for line in file:
                if not line.strip():
                    continue
                json_data = json.loads(line)
                text = json_data["text"]
                concatenated_text += text + "\n"
        concatenated_text = concatenated_text.rstrip("\n")
        return concatenated_text

    def __len__(self) -> int:
        return len(self.song_pathes)

    def __getitem__(self, idx):
        file_path_pair = self.song_pathes[idx]
        file_path = Path(file_path_pair[0])
        out_path = Path(file_path_pair[1])
        state_path = Path(file_path_pair[2])
        info = file_path_pair[3:]
        assert file_path.exists(), f"Data path {file_path} does not exist."
        assert file_path.is_file(), f"Data path {file_path} is not a file."
        lrc = self.readfile(file_path)
        return {
            "file_path": file_path,
            "out_path": out_path,
            "state_path": state_path,
            "file_group": idx // self.output_group_size,
            "lrc_data": lrc,
            "info": info
        }

    @staticmethod
    def collate_fn(batch: list):
        assert len(batch) == 1
        return batch[0]