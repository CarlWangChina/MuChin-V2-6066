from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.hparams import get_hparams
from .tokenizer import LyricsTokenizer
from .tiktoken import TikTokenLyricsTokenizer

_lyrics_tokenizer = None

def get_lyrics_tokenizer(hparams=None) -> LyricsTokenizer:
    global _lyrics_tokenizer
    if _lyrics_tokenizer is None:
        if hparams is None:
            hparams = get_hparams()
        name = hparams["ama-prof-divi"]["models"]["lyrics"]["tokenizer"]["name"]
        if name == "tiktoken":
            _lyrics_tokenizer = TikTokenLyricsTokenizer(hparams)
        else:
            raise ValueError(f"Unknown tokenizer: {name}")
    return _lyrics_tokenizer