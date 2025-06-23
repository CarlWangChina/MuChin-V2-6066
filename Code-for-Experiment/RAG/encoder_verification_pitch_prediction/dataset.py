import torch
import os
import random
import math
import logging
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from utils import *

logger = logging.getLogger(__name__)

class Tokens2MelodyDataset(Dataset):
    def __init__(self, dirpath, pitch_start, freq=75, section_len=24):
        super().__init__()
        self.section_len = section_len
        seclen_frame = section_len * freq
        tokens_path = os.path.join(dirpath, 'tokens')
        midi_path = os.path.join(dirpath, 'midi')
        self.all_tokens = []
        self.all_melody = []
        dirlist = os.listdir(tokens_path)
        print('Creating dataset...')
        for filename in tqdm(dirlist, dynamic_ncols=False):
            filepath = os.path.join(tokens_path, filename)
            tokens = np.load(filepath)
            midi_file = filename[:-17] + '.mp3_5b.mid'
            midi_file_path = os.path.join(midi_path, midi_file)
            midi = LeadSheet(midi_file_path)
            if midi.melody is not None:
                melody = midi.melody.onset_representation(freq)
                assert len(melody) <= len(tokens)
                tokens = tokens[:len(melody)]
                self.all_melody.append(melody)
                self.all_tokens.append(tokens)
        assert len(self.all_tokens) == len(self.all_melody)
        self.all_sec_tokens = []
        self.all_sec_melody = []
        for i in range(len(self.all_tokens)):
            tokens = self.all_tokens[i]
            melody = self.all_melody[i]
            if len(tokens) >= seclen_frame:
                for sec in range(0, len(tokens), seclen_frame):
                    if len(tokens) - sec >= seclen_frame:
                        self.all_sec_tokens.append(tokens[sec: sec + seclen_frame])
                        self.all_sec_melody.append(melody[sec: sec + seclen_frame])
        self.all_sec_melody = np.array(self.all_sec_melody)
        self.all_sec_tokens = np.array(self.all_sec_tokens)
        self.all_sec_melody[self.all_sec_melody > 0] -= pitch_start

    def __len__(self):
        return len(self.all_sec_tokens)

    def __getitem__(self, idx):
        tokens = self.all_sec_tokens[idx]
        melody = self.all_sec_melody[idx]
        return tokens, melody

class Tokens2MelodyDataLoader(DataLoader):
    def __init__(self, dataset, config, sampler=None):
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers'] if 'num_workers' in config.keys() else 0
        self.shuffle = config['shuffle']
        self.drop_last = config["drop_last"]
        self.batch_size = min(self.batch_size, len(dataset))
        kwarg = {
            'dataset': dataset,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'shuffle': self.shuffle,
            'drop_last': self.drop_last,
            'pin_memory': True
        }
        if sampler is not None:
            kwarg["sampler"] = sampler
            kwarg["shuffle"] = False
        super().__init__(**kwarg)

def get_data(config):
    train_config = config["train"]
    valid_config = config["valid"]
    test_config = config["test"]
    p_train = train_config["proportion"]
    p_valid = valid_config["proportion"]
    p_test = test_config["proportion"]
    assert p_train >= 0
    assert p_valid >= 0
    assert p_test >= 0
    assert p_train + p_valid + p_test == 1
    path = config["path"]
    freq = config["freq"]
    pitch_start = config["pitch_start"]
    section_len = config["section_len"]
    representation = config["representation"]
    if "mert_path" in config.keys():
        mert_path = config["mert_path"]
    else:
        mert_path = None
    if mert_path is not None:
        dataset = MERT2PitchDataset(path, mert_path, pitch_start, freq, section_len)
    elif representation == "onset":
        dataset = Tokens2MelodyDataset(path, pitch_start, freq, section_len)
    elif representation == "pitch_onset" or representation == "PitchOnset":
        dataset = Tokens2PitchOnsetDataset(path, pitch_start, freq, section_len)
    else:
        raise ValueError()
    generator = torch.Generator().manual_seed(6)
    dataset_size = len(dataset)
    logger.debug(f"Num of data: {dataset_size}")
    train_size = math.floor(dataset_size * p_train)
    valid_size = math.floor(dataset_size * p_valid)
    test_size = dataset_size - train_size - valid_size
    logger.debug(f"{train_size} {valid_size} {test_size}")
    train_set, valid_set, test_set = random_split(dataset, [train_size, valid_size, test_size], generator=generator)
    return train_set, valid_set, test_set

class Tokens2PitchOnsetDataset(Dataset):
    def __init__(self, dirpath, pitch_start, freq=75, section_len=24):
        super().__init__()
        self.section_len = section_len
        seclen_frame = section_len * freq
        tokens_path = os.path.join(dirpath, 'tokens')
        midi_path = os.path.join(dirpath, 'midi')
        self.all_tokens = []
        self.all_pitches = []
        self.all_onset = []
        dirlist = os.listdir(tokens_path)
        print('Creating dataset...')
        for filename in tqdm(dirlist):
            filepath = os.path.join(tokens_path, filename)
            tokens = np.load(filepath)
            midi_file = filename[:-17] + '.mp3_5b.mid'
            midi_file_path = os.path.join(midi_path, midi_file)
            midi = LeadSheet(midi_file_path)
            if midi.melody is not None:
                melody = midi.melody.pitch_onset_repr(freq)
                pitches, is_onset = melody
                assert len(pitches) <= len(tokens)
                tokens = tokens[:len(pitches)]
                self.all_pitches.append(pitches)
                self.all_onset.append(is_onset)
                self.all_tokens.append(tokens)
        assert len(self.all_tokens) == len(self.all_pitches)
        self.all_sec_tokens = []
        self.all_sec_pitches = []
        self.all_sec_onset = []
        for i in range(len(self.all_tokens)):
            tokens = self.all_tokens[i]
            pitches = self.all_pitches[i]
            is_onset = self.all_onset[i]
            assert len(pitches) == len(is_onset)
            if len(tokens) >= seclen_frame:
                for sec in range(0, len(tokens), seclen_frame):
                    if len(tokens) - sec >= seclen_frame:
                        self.all_sec_tokens.append(tokens[sec: sec + seclen_frame])
                        self.all_sec_pitches.append(pitches[sec: sec + seclen_frame])
                        self.all_sec_onset.append(is_onset[sec: sec + seclen_frame])
        self.all_sec_pitches = np.array(self.all_sec_pitches, dtype=np.int64)
        self.all_sec_tokens = np.array(self.all_sec_tokens, dtype=np.int64)
        self.all_sec_onset = np.array(self.all_sec_onset, dtype=np.int64)
        self.all_sec_pitches[self.all_sec_pitches > 0] -= pitch_start

    def __len__(self):
        return len(self.all_sec_tokens)

    def __getitem__(self, idx):
        tokens = self.all_sec_tokens[idx]
        pitches = self.all_sec_pitches[idx]
        is_onset = self.all_sec_onset[idx]
        return tokens, pitches, is_onset

class TestPitchDataset(Tokens2PitchOnsetDataset):
    def __init__(self, dirpath, pitch_start, freq=75, section_len=24):
        super().__init__()
        self.section_len = section_len
        seclen_frame = section_len * freq
        tokens_path = os.path.join(dirpath, 'tokens')
        midi_path = os.path.join(dirpath, 'midi')
        self.all_tokens = []
        self.all_pitches = []
        self.all_onset = []
        dirlist = os.listdir(tokens_path)
        random.seed(53)
        random.shuffle(dirlist)
        print('Creating dataset...')
        for filename in tqdm(dirlist[:100]):
            filepath = os.path.join(tokens_path, filename)
            tokens = np.load(filepath)
            midi_file = filename[:-17] + '.mp3_5b.mid'
            midi_file_path = os.path.join(midi_path, midi_file)
            midi = LeadSheet(midi_file_path)
            if midi.melody is not None:
                melody = midi.melody.pitch_onset_repr(freq)
                pitches, is_onset = melody
                assert len(pitches) <= len(tokens)
                tokens = tokens[:len(pitches)]
                self.all_pitches.append(pitches)
                self.all_onset.append(is_onset)
                self.all_tokens.append(tokens)
        assert len(self.all_tokens) == len(self.all_pitches)
        self.all_sec_tokens = []
        self.all_sec_pitches = []
        self.all_sec_onset = []
        for i in range(len(self.all_tokens)):
            tokens = self.all_tokens[i]
            pitches = self.all_pitches[i]
            is_onset = self.all_onset[i]
            assert len(pitches) == len(is_onset)
            if len(tokens) >= seclen_frame:
                for sec in range(0, len(tokens), seclen_frame):
                    if len(tokens) - sec >= seclen_frame:
                        self.all_sec_tokens.append(tokens[sec: sec + seclen_frame])
                        self.all_sec_pitches.append(pitches[sec: sec + seclen_frame])
                        self.all_sec_onset.append(is_onset[sec: sec + seclen_frame])
        self.all_sec_pitches = np.array(self.all_sec_pitches, dtype=np.int64)
        self.all_sec_tokens = np.array(self.all_sec_tokens, dtype=np.int64)
        self.all_sec_onset = np.array(self.all_sec_onset, dtype=np.int64)
        self.all_sec_pitches[self.all_sec_pitches > 0] -= pitch_start

    def __len__(self):
        return len(self.all_sec_tokens)

    def __getitem__(self, idx):
        tokens = self.all_sec_tokens[idx]
        pitches = self.all_sec_pitches[idx]
        is_onset = self.all_sec_onset[idx]
        return tokens, pitches, is_onset

class MERT2PitchDataset(Dataset):
    def __init__(self, dirpath, mert_path, pitch_start, freq=75, section_len=24):
        super().__init__()
        self.section_len = section_len
        seclen_frame = section_len * freq
        midi_path = os.path.join(dirpath, 'midi')
        self.all_merts = []
        self.all_pitches = []
        self.all_onset = []
        print('Creating dataset...')
        dirlist = os.listdir(mert_path)
        for filename in tqdm(dirlist[:1000]):
            midi_file = filename.split("_")[0] + '.mp3_5b.mid'
            midi_file_path = os.path.join(midi_path, midi_file)
            if not os.path.exists(midi_file_path):
                continue
            filepath = os.path.join(mert_path, filename)
            with open(filepath, 'rb') as f:
                mert = f.read()
                mert = np.frombuffer(mert, dtype=np.float32).reshape(-1, 1024)
            midi = LeadSheet(midi_file_path)
            if midi.melody is not None:
                melody = midi.melody.pitch_onset_repr(freq)
                pitches, is_onset = melody
                cut_idx = min(len(pitches), len(mert))
                pitches = pitches[:cut_idx]
                is_onset = is_onset[:cut_idx]
                mert = mert[:cut_idx]
                self.all_pitches.append(pitches)
                self.all_onset.append(is_onset)
                self.all_merts.append(mert)
        assert len(self.all_merts) == len(self.all_pitches)
        self.all_sec_merts = []
        self.all_sec_pitches = []
        self.all_sec_onset = []
        logger.debug("Section start.")
        for i in range(len(self.all_merts)):
            mert = self.all_merts[i]
            pitches = self.all_pitches[i]
            is_onset = self.all_onset[i]
            assert len(pitches) == len(is_onset)
            if len(mert) >= seclen_frame:
                for sec in range(0, len(mert), seclen_frame):
                    if len(mert) - sec >= seclen_frame:
                        self.all_sec_merts.append(mert[sec: sec + seclen_frame])
                        self.all_sec_pitches.append(pitches[sec: sec + seclen_frame])
                        self.all_sec_onset.append(is_onset[sec: sec + seclen_frame])
        self.all_sec_pitches = np.array(self.all_sec_pitches, dtype=np.int64)
        self.all_sec_onset = np.array(self.all_sec_onset, dtype=np.int64)
        self.all_sec_pitches[self.all_sec_pitches > 0] -= pitch_start
        logger.debug("Section done.")

    def __len__(self):
        return len(self.all_sec_merts)

    def __getitem__(self, idx):
        tokens = self.all_sec_merts[idx]
        tokens = np.array(tokens)
        pitches = self.all_sec_pitches[idx]
        is_onset = self.all_sec_onset[idx]
        return tokens, pitches, is_onset

def get_test_data():
    dataset = TestPitchDataset("data", 21)
    trainloader = DataLoader(dataset, 2)
    validloader = trainloader
    testloader = trainloader
    return trainloader, validloader, testloader

if __name__ == '__main__':
    dirpath = './data'
    my_dataset = MERT2PitchDataset(dirpath, "/data/xary/mert", 21)
    print(len(my_dataset))
    exit()

import yaml
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "-c",
    "--config",
    type=str,
    help="path of the yaml format config file"
)
parser.set_defaults(
    config="./config/config.yaml"
)
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
    print(config)
data_config = config["data"]
model_config = config["model"]
trainloader, validloader, testloader = get_data(data_config)
for data in trainloader:
    tokens, pitches, is_onset = data
    print(tokens.shape, tokens)
    print(pitches.shape, pitches)
    print(is_onset)
    break