from ama_prof_divi.configs import get_hparams
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.models.acoustic.encodec.encodec import EncodecModelWrapper

_wrapper = None

def get_encodec_wrapper(hparams: dict = None) -> EncodecModelWrapper:
    global _wrapper
    if _wrapper is None:
        if hparams is None:
            hparams = get_hparams()
        _wrapper = EncodecModelWrapper(hparams)
    return _wrapper