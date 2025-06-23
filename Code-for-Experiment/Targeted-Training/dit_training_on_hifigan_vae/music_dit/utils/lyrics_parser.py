import re
import opencc
import string
import torch
from dataclasses import dataclass
from typing import List
from pathlib import Path
from .hparams import get_hparams
from .logging import get_logger
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.modules.tiktoken.tiktoken import TikTokenWrapper

logger = get_logger(__name__)

@dataclass
class Lyrics:
    text: str
    tokens: torch.Tensor
    start_time: float
    end_time: float

    def __repr__(self):
        return f"Lyrics(\"{self.text}\", start_time={self.start_time}, end_time={self.end_time})"

    def cat(self, *others: 'Lyrics') -> 'Lyrics':
        text = self.text
        tokens = self.tokens
        start_time = self.start_time
        end_time = self.end_time
        for other in others:
            text += (" " + other.text)
            tokens = torch.cat([tokens, other.tokens], dim=0)
            end_time = other.end_time
        return Lyrics(text=text,
                      tokens=tokens,
                      start_time=start_time,
                      end_time=end_time)

class LyricFileParser:
    def __init__(self):
        self.opencc_converter = opencc.OpenCC('t2s')
        self.matcher = re.compile(r"Text:(.*?) Timestamps:\((\d+(?:\.\d+)?), (\d+(?:\.\d+)?)\)")
        self.tokenizer = TikTokenWrapper()
        self.device = get_hparams().device
        self.filter_pattern = re.compile(f"^[0-9\s\-{re.escape(string.punctuation)}]*$")
        self.hallucinations = []
        hallucination_table_path = Path(__file__).parent.joinpath('hallucinations.txt')
        with open(hallucination_table_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line == "" or line.startswith(""):
                    continue
                self.hallucinations.append(line)

    def parse(self,
              lyric_file: any) -> List[Lyrics]:
        result = []
        with open(lyric_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                match = self.matcher.match(line)
                if match is not None:
                    text = match.group(1)
                    text = self.opencc_converter.convert(text).lower()
                    text = self._filter_lyrics_text(text)
                    if text == "":
                        continue
                    start_time = float(match.group(2))
                    end_time = float(match.group(3))
                    tokens = self.tokenizer.encode(text)
                    result.append(Lyrics(text=text,
                                         tokens=torch.Tensor(tokens).long().to(self.device),
                                         start_time=start_time,
                                         end_time=end_time))
        return result

    def _filter_lyrics_text(self,
                            text: str) -> str:
        text = re.sub(r'\(.*?\)', '', text)
        for hallucination in self.hallucinations:
            if hallucination.lower() in text:
                return ""
        if self.filter_pattern.match(text):
            return ""
        return text.strip()