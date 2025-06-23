import os
import random
import torch
import torch.utils.data
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator

class AudioDatasetOnline(torch.utils.data.Dataset):
    def __init__(self, *, wav_roots, chunk_size=128000, sample_rate=16000, min_duration=1.0):
        self.sampling_rate = sample_rate
        self.chunk_size = chunk_size
        self.min_duration = min_duration
        self.max_duration = chunk_size / sample_rate
        self.datas = []
        audio_files = []
        assert type(wav_roots) == list or type(wav_roots) == str, "wav_roots should be a list or a string"
        if type(wav_roots) == str:
            wav_roots = [wav_roots]
        for wav_root in wav_roots:
            for path, _, files in os.walk(wav_root):
                for file in files:
                    if file.endswith('.wav') or file.endswith('.mp3'):
                        audio_files.append(os.path.join(path, file))
        audio_files = list(set(audio_files))
        random.shuffle(audio_files)
        self.datas = audio_files
        assert len(self.datas) > 1, "No Valid Audio Files Contained"

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

    def __getitem__(self, index):
        try:
            wav, sr = torchaudio.load(self.datas[index])
            wav_16k = torchaudio.transforms.Resample(sr, self.sampling_rate)(wav)
            wav_mask = torch.ones(self.chunk_size).bool()
            assert len(wav_16k.size()) == 2 or len(wav_16k.size()) == 1, f"Audio {self.datas[index]} is not mono audio"
            if wav_16k.size(0) == 2:
                wav_16k = wav_16k.mean(0, keepdim=True)
            if wav_16k.size(1) < self.chunk_size:
                wav_16k = F.pad(wav_16k, (0, self.chunk_size - wav_16k.size(1)), 'constant', 0)
                wav_mask[wav_16k.size(1):] = True
            else:
                start_point = random.randint(0, wav_16k.size(1) - self.chunk_size)
                wav_16k = wav_16k[:, start_point:start_point + self.chunk_size]
        except Exception as e:
            wav_16k = torch.zeros(1, self.chunk_size)
            wav_mask = torch.zeros(self.chunk_size).bool()
            print(f"Error: {e}")
        return wav_16k, wav_mask

    def __len__(self):
        return len(self.datas)

    @staticmethod
    def make_datasets(
        train_portion=0.9,
        data_pths=['datas/audio-original-mp3', 'datas/audio-separate-vocal', 'datas/gansheng', 'datas/svc-opensource-dataset'],
        chunk_size=128000, sample_rate=16000, min_duration=1.0,
    ):
        audio_dataset = AudioDatasetOnline(wav_roots=data_pths, chunk_size=chunk_size, sample_rate=sample_rate, min_duration=min_duration)
        train_size = int(len(audio_dataset) * train_portion)
        eval_size = len(audio_dataset) - train_size
        train_dataset, eval_dataset = torch.utils.data.random_split(audio_dataset, [train_size, eval_size], generator=torch.Generator().manual_seed(3507))
        return train_dataset, eval_dataset

if __name__ == '__main__':
    datas = ['datas/audio-original-mp3', 'datas/audio-separate-vocal', 'datas/gansheng', 'datas/svc-opensource-dataset']
    audio_dataset = AudioDatasetOnline(wav_roots=datas)
    audio_loader = torch.utils.data.DataLoader(audio_dataset, batch_size=4, shuffle=True, num_workers=8)
    for i, (wav_padded, wav_mask) in enumerate(audio_loader):
        print(wav_padded.shape, wav_mask.sum(dim=-1))
        if i == 10:
            break
    print("Done")