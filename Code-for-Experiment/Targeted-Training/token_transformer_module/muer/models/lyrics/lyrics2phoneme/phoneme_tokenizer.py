from pathlib import Path
import torch.nn as nn
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.logging import get_logger
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.models.lyrics.builder import get_lyrics_tokenizer
logger = get_logger(__name__)
ENGLISH_PHONEMES = ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']

PINYIN_TO_PHONEME_DICT_FILE = "pinyin2phoneme.txt"

def _get_chinese_phonemes():
    chinese_phonemes = set()
    with open(Path(__file__).parent / PINYIN_TO_PHONEME_DICT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            words = line.split()
            words = [w.strip().lower() for w in words]
            if len(words) >= 2:
                ph = words[1:]
                chinese_phonemes.update(ph)
    return list(chinese_phonemes)

class PhonemeTokenizer(nn.Module):
    def __init__(self, hparams: dict):
        super(PhonemeTokenizer, self).__init__()
        self.hparams = hparams
        self.tok_hparams = self.hparams["ama-prof-divi"]["models"]["lyrics"]["tokenizer"]
        self.pad_token = self.tok_hparams["pad_token"]
        self.start_token = self.tok_hparams["start_token"]
        self.end_token = self.tok_hparams["end_token"]
        self.mask_token = self.tok_hparams["mask_token"]
        self.unknown_token = self.tok_hparams["unknown_token"]
        self.sep_token = self.tok_hparams["sep_token"]
        english_phonemes = ENGLISH_PHONEMES
        english_phonemes.sort()
        chinese_phonemes = _get_chinese_phonemes()
        chinese_phonemes.sort()
        lyrics_tokenizer = get_lyrics_tokenizer(hparams)
        special_tokens = list(lyrics_tokenizer.special_tokens_set())
        special_tokens.sort()
        idx = 0
        self.phoneme_dict = {}
        self.phoneme_list = []
        for token in english_phonemes:
            self.phoneme_list.append(token)
            self.phoneme_dict[token] = idx
            idx += 1
        for token in chinese_phonemes:
            self.phoneme_list.append(token)
            self.phoneme_dict[token] = idx
            idx += 1
        for token in special_tokens:
            self.phoneme_list.append(token)
            self.phoneme_dict[token] = idx
            idx += 1
        assert idx == len(self.phoneme_list)
        assert idx == len(self.phoneme_dict)
        logger.info("Phoneme dictionary: vocab size = %d", self.vocab_size)
        logger.info("Pad id = %d", self.pad_id)
        logger.info("Sep id = %d", self.sep_id)

    @property
    def vocab_size(self):
        return len(self.phoneme_dict)

    @property
    def pad_id(self):
        return self.phoneme_dict[self.pad_token]

    @property
    def start_id(self):
        return self.phoneme_dict[self.start_token]

    @property
    def end_id(self):
        return self.phoneme_dict[self.end_token]

    @property
    def mask_id(self):
        return self.phoneme_dict[self.mask_token]

    @property
    def unknown_id(self):
        return self.phoneme_dict[self.unknown_token]

    @property
    def sep_id(self):
        return self.phoneme_dict[self.sep_token]

    def encode(self, phonemes: str) -> [int]:
        phonemes = phonemes.strip().split()
        phoneme_ids = []
        for p in phonemes:
            if p in self.phoneme_dict:
                phoneme_ids.append(self.phoneme_dict[p])
            else:
                raise ValueError("Unknown phoneme: {}".format(p))
        return phoneme_ids

    def decode(self, phoneme_ids: [int]) -> str:
        phonemes = []
        for pid in phoneme_ids:
            assert 0 <= pid < self.vocab_size, "Invalid phoneme id: {}".format(pid)
            phonemes.append(self.phoneme_list[pid])
        return " ".join(phonemes)