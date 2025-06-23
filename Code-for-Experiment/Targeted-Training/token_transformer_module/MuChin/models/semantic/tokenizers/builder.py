from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.hparams import get_hparams
from .kmeans import KMeansTokenizer
from .rvq import RVQTokenizer

_tokenizer = None
_melody_tokenizer = None

def get_tokenizer(hparams: dict = None) -> KMeansTokenizer or RVQTokenizer:
    if hparams is None:
        hparams = get_hparams()
    tokenizer_name = hparams["ama-prof-divi"]["models"]["semantic"]["tokenizer"]["name"]
    if tokenizer_name == "kmeans":
        return KMeansTokenizer(hparams, load_pretrained=False)
    elif tokenizer_name == "rvq":
        return RVQTokenizer(hparams, load_pretrained=False)
    else:
        raise ValueError("Unsupported tokenizer name: {}".format(tokenizer_name))

def get_melody_tokenizer(hparams: dict = None) -> KMeansTokenizer or RVQTokenizer:
    global _melody_tokenizer
    if _melody_tokenizer is None:
        if hparams is None:
            hparams = get_hparams()
        tokenizer_name = hparams["ama-prof-divi"]["models"]["semantic"]["tokenizer"]["name"]
        if tokenizer_name == "kmeans":
            _melody_tokenizer = KMeansTokenizer(hparams, load_pretrained=True)
        elif tokenizer_name == "rvq":
            _melody_tokenizer = RVQTokenizer(hparams, load_pretrained=True)
        else:
            raise ValueError("Unsupported tokenizer name: {}".format(tokenizer_name))
    return _melody_tokenizer