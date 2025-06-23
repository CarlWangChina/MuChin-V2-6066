import os
import time
import json
import re
import Code_for_Experiment.Metrics.music_understanding_model.tensorboardX.tests.test_numpy
import gzip
from tqdm import tqdm
from collections import defaultdict
from ama_prof_divi.configs import init_hparams, get_hparams, post_init_hparams
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.models.lyrics.builder import get_lyrics_tokenizer
from ama_prof_divi.utils import logging
import numpy as np

class SongStructureInsertor:
    def __init__(self, hparams, mode='melody') -> None:
        self.lyrics_tokenizer = get_lyrics_tokenizer(hparams)
        self.hparams = hparams
        self.special_tokens_dict = {}

        self.vocab_size = self.hparams["ama-prof-divi"]["models"]["semantic"]["melody_tokenizer"]["num_clusters"]

        self.special_tokens_dict = defaultdict(int)
        for token in self.lyrics_tokenizer.special_tokens_set():
            self.special_tokens_dict[token] = self.vocab_size
            self.vocab_size += 1
        self.mode = mode

    def process(self, file_pth):
        for i, f in enumerate(file_pth):
            tokens = np.load(f)
            with open(f, 'r') as file:
                timestamp_data = json.load(file)
            inserted_data = self.inference(tokens, timestamp_data)
            np.save(f, inserted_data)

    def timestamp_to_seconds(self, timestamp):
        minutes, seconds = map(float, timestamp.strip('[]').split(':'))
        return 60 * minutes + seconds

    def extract_special_token(self, text):
        match = re.search(r'(<.*?>)', text)
        return match.group(1) if match else None

    def inference(self, tokens, timestamp_data):
        insertions = []

        for timestamp, text in timestamp_data.items():
            special_token_text = self.extract_special_token(text)
            if special_token_text and special_token_text in self.special_tokens_dict:
                seconds = self.timestamp_to_seconds(timestamp)
                insert_position = int(seconds * 75) if self.mode == 'melody' else int(seconds * 5)
                token_to_insert = self.special_tokens_dict[special_token_text]

                insertions.append((insert_position, token_to_insert))
        insertions.sort(reverse=True)
        for position, token in insertions:
            tokens = np.insert(tokens, position, token)
        return tokens

if __name__ == '__main__':
    init_hparams()
    post_init_hparams()
    hparams = get_hparams()
    trainer = SongStructureInsertor(hparams)
    file_pth = ...
    trainer.process(file_pth)