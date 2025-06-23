from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.hparams import get_hparams
from .svs import SVSModel

_svs_model = None

def get_svs_model(hparams=None):
    global _svs_model
    if _svs_model is None:
        if hparams is None:
            hparams = get_hparams()
        _svs_model = SVSModel(hparams)
    return _svs_model