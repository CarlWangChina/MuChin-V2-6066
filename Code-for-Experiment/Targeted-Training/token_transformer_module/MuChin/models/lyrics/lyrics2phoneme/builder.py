from ama_prof_divi.configs import get_hparams
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.models.lyrics.lyrics2phoneme.phoneme_tokenizer import PhonemeTokenizer
from .lyrics2phoneme import Lyrics2PhonemeTranslator

_lyrics_to_phoneme_translator = None

def get_lyrics_to_phoneme_translator(hparams: dict = None) -> Lyrics2PhonemeTranslator:
    global _lyrics_to_phoneme_translator
    if _lyrics_to_phoneme_translator is None:
        if hparams is None:
            hparams = get_hparams()
        _lyrics_to_phoneme_translator = Lyrics2PhonemeTranslator(hparams)
    return _lyrics_to_phoneme_translator

def get_phoneme_tokenizer(hparams: dict = None) -> PhonemeTokenizer:
    return get_lyrics_to_phoneme_translator(hparams).phoneme_tokenizer