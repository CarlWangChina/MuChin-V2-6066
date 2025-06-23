import tiktoken
from ama_prof_divi.utils import logging
from .tokenizer import LyricsTokenizer

logger = logging.get_logger(__name__)

class ama_prof_diviEncoding(tiktoken.Encoding):
    def __init__(self, base_encoding: tiktoken.Encoding, special_tokens: dict):
        super(ama_prof_diviEncoding, self).__init__(
            name=base_encoding.name + "_ama_prof_divi",
            pat_str=base_encoding._pat_str,
            mergeable_ranks=base_encoding._mergeable_ranks,
            special_tokens={
                **base_encoding._special_tokens,
                **special_tokens
            }
        )

class TikTokenLyricsTokenizer(LyricsTokenizer):
    def __init__(self, hparams: dict):
        super(TikTokenLyricsTokenizer, self).__init__(hparams)
        logger.info(f"Base (pretrained) model used for TikToken: {self.tok_hparams['pretrained_model']}")
        cl100k_base = tiktoken.get_encoding(self.tok_hparams["pretrained_model"])
        special_token_id = cl100k_base.n_vocab
        special_tokens = {}
        for token in sorted(self.tok_hparams["special_tokens"]):
            special_tokens[token] = special_token_id
            special_token_id += 1
        self.encoding = ama_prof_diviEncoding(cl100k_base, special_tokens)
        self.special_tokens_dict = {}
        for token in sorted(self.encoding.special_tokens_set):
            self.special_tokens_dict[token] = self.encode(token)[0]
        logger.info(f"Special tokens: {self.special_tokens_dict}")

    def _get_model_name(self):
        return "tiktoken_%s" % self.encoding.name

    def _get_vocab_size(self):
        return self.encoding.n_vocab

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

    def special_tokens_set(self):
        return self.encoding.special_tokens_set

    def is_special_token(self, token: str or int) -> bool:
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