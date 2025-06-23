from ama_prof_divi.configs import get_hparams
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.models.song_structures.ss_tokenizer import SSTokenizer

_tokenizer = None

def get_ss_tokenizer(hparams: dict = None) -> SSTokenizer:
    global _tokenizer
    if _tokenizer is None:
        if hparams is None:
            hparams = get_hparams()
        _tokenizer = SSTokenizer(hparams)
    return _tokenizer