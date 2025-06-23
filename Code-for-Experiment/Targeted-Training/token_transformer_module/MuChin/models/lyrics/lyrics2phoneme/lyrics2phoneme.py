import torch
import torch.nn as nn
import re
from pathlib import Path
from ama_prof_divi.utils.logging import get_logger
from g2pM import G2pM
from g2p_en import G2p
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.modules.phoneme.tokenizer import PhonemeTokenizer
import get_lyrics_tokenizer
logger = get_logger(__name__)
PINYIN_TO_PHONEME_DICT_FILE = "pinyin2phoneme.txt"
SPECIAL_TOKEN_PATTERN = r"<\|.*?\|>"
SEP_CHARACTERS = ["，", "。", "？", "：", "！", "；", ",", ".", ":", ";", " ", "?", "!", "、", "/", "\\", "\r", "\n"]
SPACE_CHARACTERS = [" ", "　", "\t"]
CHAR_TYPE_RANGE_MAP = [
    (0x27, 0x27, "en_alphabet"),
    (0x30, 0x39, "en_digit"),
    (0x41, 0x5a, "en_alphabet"),
    (0x61, 0x7a, "en_alphabet"),
    (0x100, 0x17f, "latin_alphabet"),
    (0x180, 0x24f, "latin_alphabet"),
    (0x370, 0x3ff, "greek"),
    (0x400, 0x4ff, "cyrillic"),
    (0x530, 0x58f, "armenian"),
    (0x590, 0x5ff, "hebrew"),
    (0x600, 0x6ff, "arabic"),
    (0x700, 0x74f, "syriac"),
    (0x750, 0x77f, "arabic"),
    (0x780, 0x7bf, "thaana"),
    (0x900, 0x97f, "devanagari"),
    (0x980, 0x9ff, "bengali"),
    (0xa00, 0xa7f, "gurmukhi"),
    (0xa80, 0xaff, "gujarati"),
    (0xb00, 0xb7f, "oriya"),
    (0xb80, 0xbff, "tamil"),
    (0xc00, 0xc7f, "telugu"),
    (0xc80, 0xcff, "kannada"),
    (0xd00, 0xd7f, "malayalam"),
    (0xd80, 0xdff, "sinhala"),
    (0xe00, 0xe7f, "thai"),
    (0xe80, 0xeff, "lao"),
    (0xf00, 0xfff, "tibetan"),
    (0x1000, 0x109f, "myanmar"),
    (0x10a0, 0x10ff, "georgian"),
    (0x1100, 0x11ff, "korean"),
    (0x1200, 0x137f, "ethiopic"),
    (0x13a0, 0x13ff, "cherokee"),
    (0x1780, 0x17ff, "khmer"),
    (0x1800, 0x18af, "mongolian"),
    (0x2e80, 0x2eff, "chinese"),
    (0x2f00, 0x2fdf, "chinese"),
    (0x3000, 0x303f, "japanese"),
    (0x3040, 0x309f, "japanese"),
    (0x30a0, 0x30ff, "japanese"),
    (0x3100, 0x312f, "chinese"),
    (0x3130, 0x318f, "korean"),
    (0x31c0, 0x31ef, "japanese"),
    (0x31f0, 0x31ff, "korean"),
    (0x3400, 0x4dbf, "chinese"),
    (0x4e00, 0x9fff, "chinese"),
    (0xa960, 0xa97f, "korean"),
    (0xac00, 0xd7a3, "korean"),
    (0xd7b0, 0xd7ff, "korean"),
    (0xf900, 0xfaff, "chinese"),
    (0xfb50, 0xfdff, "arabic"),
    (0xfe70, 0xfeff, "arabic"),
    (0xfe30, 0xfe4f, "chinese"),
    (0x20000, 0x2a6df, "chinese"),
]

def _load_pinyin2phoneme_dict() -> dict:
    logger.info("Loading pinyin-to-phoneme dictionary...")
    pinyin2phoneme_dict = {}
    with open(Path(__file__).parent / PINYIN_TO_PHONEME_DICT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            words = line.split()
            words = [w.strip().lower() for w in words]
            if len(words) >= 2:
                pinyin2phoneme_dict[words[0]] = words[1:]
    return pinyin2phoneme_dict

def _get_char_type(ch: str) -> str:
    if ch in SPACE_CHARACTERS:
        return "space"
    if ch in SEP_CHARACTERS:
        return "sep"
    cp = ord(ch)
    for start, end, char_type in CHAR_TYPE_RANGE_MAP:
        if start <= cp <= end:
            return char_type
    return "ignore"

def _split_sentence(sentence: str, special_tokens: set[str], sep_token: str = "<|ss_sep|>") -> [(str, str)]:
    sub_sentences = []
    tokens_with_positions = [(match.group(), match.start()) for match in re.finditer(SPECIAL_TOKEN_PATTERN, sentence)]
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
            ch_type = _get_char_type(ch)
            if (ch_type == "en_digit") or (ch_type == "ignore"):
                pass
            elif ch_type == "space":
                if current_substr != "" and current_substr_type in ["english", "unknown_lang"]:
                    current_substr += ch
            elif (current_substr != "" and current_substr_type in ["chinese", "japanese", "korean"]):
                sub_sentences.append((current_substr.strip(), current_substr_type))
                current_substr = ""
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

class Lyrics2PhonemeTranslator(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.hparams = hparams
        self.device = hparams["ama-prof-divi"]["device"]
        self.phoneme_tokenizer = PhonemeTokenizer(hparams)
        self.pinyin2phoneme_dict = _load_pinyin2phoneme_dict()
        lyrics_tokenizer = get_lyrics_tokenizer(hparams)
        self.special_tokens = lyrics_tokenizer.special_tokens_set()
        self.g2p_zh_model = G2pM()
        self.g2p_model = G2p()

    def split_sentence(self, sentence: str) -> [(str, int)]:
        return _split_sentence(sentence, self.special_tokens)

    def split_and_translate_sentence(self, sentence: str) -> [dict]:
        sub_sentences = self.split_sentence(sentence)
        output = []
        for sub_sentence in sub_sentences:
            sub_sentence_text, sub_sentence_type = sub_sentence
            if sub_sentence_type == "special_tokens":
                output.append({"text": sub_sentence_text, "type": sub_sentence_type, "phonemes": sub_sentence_text})
            elif sub_sentence_type == "sep":
                output.append({"text": sub_sentence_text, "type": sub_sentence_type, "phonemes": sub_sentence_text})
            elif sub_sentence_type == "english":
                phonemes = self.g2p_model(sub_sentence_text)
                phonemes = " ".join(p.strip().upper() for p in phonemes)
                output.append({"text": sub_sentence_text, "type": sub_sentence_type, "phonemes": phonemes})
            elif sub_sentence_type == "chinese":
                pinyin = self.g2p_zh_model(sub_sentence_text, tone=False)
                phonemes = self._convert_pinyin_to_phoneme(pinyin)
                output.append({"text": sub_sentence_text, "type": sub_sentence_type, "phonemes": phonemes})
            else:
                raise ValueError("Unknown sentence type: {}".format(sub_sentence_type))
        return output

    def _convert_pinyin_to_phoneme(self, pinyin: [str]) -> str:
        phonemes = []
        for p in pinyin:
            if p in self.pinyin2phoneme_dict:
                phonemes.append(" ".join(self.pinyin2phoneme_dict[p]))
            else:
                raise ValueError("Unknown pinyin: {}".format(p))
        return " ".join(phonemes)

    def forward(self, lyrics: [str], pad_id: int = -1) -> torch.Tensor:
        if pad_id < 0:
            pad_id = self.phoneme_tokenizer.pad_id
        num_batches = len(lyrics)
        phoneme_tokens = []
        for i in range(num_batches):
            split_and_translate_sentence = self.split_and_translate_sentence(lyrics[i])
            toks = []
            for s in split_and_translate_sentence:
                phonemes = s["phonemes"]
                toks.extend(self.phoneme_tokenizer.encode(phonemes))
            phoneme_tokens.append(toks)
        max_token_len = max(len(t) for t in phoneme_tokens)
        result = torch.full((num_batches, max_token_len), pad_id, dtype=torch.long, device=self.device)
        for i, t in enumerate(phoneme_tokens):
            result[i, :len(t)] = torch.tensor(t, dtype=torch.long, device=self.device)
        return result