import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.amp.rnn_compat
import tiktoken
from typing import Dict, Union
import torch.nn as nn
from music_dit.utils import get_logger, get_hparams

logger = get_logger(__name__)

class TikTokenWrapper(nn.Module):
    def __init__(self):
        super(TikTokenWrapper, self).__init__()
        hparams = get_hparams()
        self.base_encoding = tiktoken.get_encoding(hparams.tiktoken.pretrained_model)
        special_token_id = self.base_encoding.n_vocab
        special_tokens = {}
        for token in sorted(hparams.tiktoken.special_tokens):
            special_tokens[token] = special_token_id
            special_token_id += 1
        self.encoding = TikTokenWrapper.MyEncoding(self.base_encoding, special_tokens)
        self.special_tokens_dict = {}
        for token in sorted(self.encoding.special_tokens_set):
            self.special_tokens_dict[token] = self.encode(token)[0]
        self.PAD = "<|ss_pad|>"
        self.PAD_Token = self.encode_special_token(self.PAD)
        self.UNK = "<|ss_unk|>"
        self.UNK_Token = self.encode_special_token(self.UNK)
        self.SEP = "<|ss_sep|>"
        self.SEP_Token = self.encode_special_token(self.SEP)

    class MyEncoding(tiktoken.Encoding):
        def __init__(self, base_encoding: tiktoken.Encoding, special_tokens: Dict[str, int]):
            super(TikTokenWrapper.MyEncoding, self).__init__(
                name=base_encoding.name + "_with_special_tokens",
                pat_str=base_encoding._pat_str,
                mergeable_ranks=base_encoding._mergeable_ranks,
                special_tokens={
                    **base_encoding._special_tokens,
                    **special_tokens
                }
            )

    @property
    def special_tokens_set(self):
        return self.encoding.special_tokens_set

    @property
    def n_vocab(self):
        return self.encoding.n_vocab

    def is_special_token(self, token: Union[str, int]) -> bool:
        if isinstance(token, str):
            return token in self.encoding.special_tokens_set
        elif isinstance(token, int):
            return token in self.special_tokens_dict.values()
        else:
            raise TypeError("token must be either str or int.")

    def encode_special_token(self, token: str) -> int:
        if token not in self.special_tokens_dict:
            raise ValueError(f"{token} is not a special token.")
        return self.special_tokens_dict[token]

    def forward(self, text: str) -> list:
        return self.encode(text)

    def encode(self, text: str) -> list:
        return self.encoding.encode(text, allowed_special="all")

    def encode_batch(self, text: list[str], *, num_threads: int = 8) -> list[list[int]]:
        return self.encoding.encode_batch(text, num_threads=num_threads, allowed_special="all")

    def decode(self, token_ids: list, *, errors: str = "replace") -> str:
        return self.encoding.decode(token_ids, errors=errors)

    def decode_batch(self, tokens: list[list[int]], *, errors: str = "replace", num_threads: int = 8) -> list[str]:
        return self.encoding.decode_batch(tokens, errors=errors, num_threads=num_threads)