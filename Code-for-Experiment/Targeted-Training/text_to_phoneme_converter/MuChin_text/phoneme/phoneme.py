import re
from g2pM import G2pM
from g2p_en import G2p
from pathlib import Path
from typing import Set, Optional, Tuple, List
from omegaconf import OmegaConf
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger
from .char_types import CHAR_TYPE_RANGE_MAP

logger = get_logger(__name__)

class Lyrics2Phoneme:
    def __init__(self):
        configs_dir = Path(__file__).parent.parent.parent / "configs"
        config_file = configs_dir / "lyrics2phoneme.yaml"
        self.hparams = OmegaConf.load(config_file).lyrics2phoneme
        self.g2p_zh_model = G2pM()
        self.g2p_model = G2p()
        self.special_token_pattern = re.compile(self.hparams.special_token_pattern)
        self.pinyin_to_phoneme_dict = {}
        with open(configs_dir / self.hparams.pinyin2phoneme.dict_file, "r") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                words = line.split()
                words = [w.strip().lower() for w in words]
                if len(words) >= 2:
                    self.pinyin_to_phoneme_dict[words[0]] = words[1:]
        self.special_tokens = set()

    def _get_char_type(self, ch: str) -> str:
        if ch in self.hparams.spaces:
            return "space"
        if ch in self.hparams.separators:
            return "sep"
        cp = ord(ch)
        for start, end, char_type in CHAR_TYPE_RANGE_MAP:
            if start <= cp <= end:
                return char_type
        return "ignore"

    def _convert_pinyin_to_phoneme(self, pinyin: [str]) -> str:
        phonemes = []
        for p in pinyin:
            if p in self.pinyin_to_phoneme_dict:
                phonemes.append(" ".join(self.pinyin_to_phoneme_dict[p]))
            else:
                logger.warning("Unknown pinyin: %s", p)
        return " ".join(phonemes)

    def split_sentence(self, sentence: str, special_tokens: Optional[Set[str]] = None, sep_token: str = "<|ss_sep|>") -> List[Tuple[str, str]]:
        sub_sentences = []
        if special_tokens is None:
            special_tokens = set()
        tokens_with_positions = [(match.group(), match.start()) for match in re.finditer(self.special_token_pattern, sentence)]
        special_token_tags = [None] * len(sentence)
        for token, pos in tokens_with_positions:
            token = token.lower()
            if token in special_tokens:
                special_token_tags[pos] = (token, "special_tokens", len(token))
            else:
                raise ValueError("Unknown special token: '{}'".format(token))
        current_substr = ""
        current_substr_type = "unknown"
        current_pos = 0
        while current_pos < len(sentence):
            if special_token_tags[current_pos] is not None:
                token, s_type, token_len = special_token_tags[current_pos]
                if current_substr != "":
                    sub_sentences.append((current_substr.strip(), current_substr_type))
                    current_substr = ""
                if token == sep_token:
                    sub_sentences.append((token, "sep"))
                else:
                    sub_sentences.append((token, "special_tokens"))
                current_pos += token_len
            else:
                ch = sentence[current_pos]
                ch_type = self._get_char_type(ch)
                if (ch_type == "en_digit") or (ch_type == "ignore"):
                    pass
                elif ch_type == "space":
                    if current_substr != "" and current_substr_type in ["english", "unknown_lang"]:
                        current_substr += ch
                    elif (current_substr != "" and current_substr_type in ["chinese", "japanese", "korean"]):
                        sub_sentences.append((current_substr.strip(), current_substr_type))
                        current_substr = ""
                    else:
                        pass
                elif ch_type == "en_alphabet":
                    if current_substr == "" or current_substr_type == "english":
                        current_substr += ch
                    else:
                        sub_sentences.append((current_substr.strip(), current_substr_type))
                        current_substr = ch
                    current_substr_type = "english"
                elif ch_type == "latin_alphabet":
                    if current_substr == "" or current_substr_type == "unknown_latin_lang":
                        current_substr += ch
                    else:
                        sub_sentences.append((current_substr.strip(), current_substr_type))
                        current_substr = ch
                    current_substr_type = "unknown_latin_lang"
                elif ch_type == "chinese":
                    if current_substr == "" or current_substr_type == "chinese":
                        current_substr += ch
                    else:
                        sub_sentences.append((current_substr.strip(), current_substr_type))
                        current_substr = ch
                    current_substr_type = "chinese"
                elif ch_type == "japanese":
                    if current_substr == "" or current_substr_type == "japanese":
                        current_substr += ch
                    else:
                        sub_sentences.append((current_substr.strip(), current_substr_type))
                        current_substr = ch
                    current_substr_type = "japanese"
                elif ch_type == "korean":
                    if current_substr == "" or current_substr_type == "korean":
                        current_substr += ch
                    else:
                        sub_sentences.append((current_substr.strip(), current_substr_type))
                        current_substr = ch
                    current_substr_type = "korean"
                elif ch_type == "mongolian":
                    if current_substr == "" or current_substr_type == "mongolian":
                        current_substr += ch
                    else:
                        sub_sentences.append((current_substr.strip(), current_substr_type))
                        current_substr = ch
                    current_substr_type = "mongolian"
                elif ch_type == "sep":
                    if current_substr != "" and current_substr_type != "sep":
                        sub_sentences.append((current_substr.strip(), current_substr_type))
                    current_substr = sep_token
                    current_substr_type = "sep"
                else:
                    if current_substr == "" or current_substr_type == "unknown_lang":
                        current_substr += ch
                    else:
                        sub_sentences.append((current_substr.strip(), current_substr_type))
                        current_substr = ch
                    current_substr_type = "unknown_lang"
                current_pos += 1
        if current_substr != "":
            sub_sentences.append((current_substr.strip(), current_substr_type))
        return sub_sentences

    def translate(self, sentence: str) -> str:
        phonemes = []
        sub_sentences = self.split_sentence(sentence, special_tokens=self.special_tokens)
        for sub_sentence, sub_sentence_type in sub_sentences:
            if sub_sentence_type == "special_tokens":
                phonemes.append(sub_sentence)
            elif sub_sentence_type == "sep":
                phonemes.append(sub_sentence)
            elif sub_sentence_type == "english":
                pm = self.g2p_model(sub_sentence)
                pmt = []
                for p in pm:
                    p = p.strip().upper()
                    if p != "":
                        pmt.append("E." + p)
                pm = " ".join(pmt)
                phonemes.append(pm)
            elif sub_sentence_type == "chinese":
                pinyin = self.g2p_zh_model(sub_sentence, tone=False)
                pm = self._convert_pinyin_to_phoneme(pinyin).split(" ")
                pmt = []
                for p in pm:
                    p = p.strip().upper()
                    if p != "":
                        pmt.append("C." + p)
                pm = " ".join(pmt)
                phonemes.append(pm)
            else:
                pass
        return " ".join(phonemes)