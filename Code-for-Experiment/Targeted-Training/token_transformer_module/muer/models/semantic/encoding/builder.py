from ama_prof_divi.configs import get_hparams
from .encoder import SemanticEncoder
import MertEncoder_semantic

_semantic_encoder = None

def get_semantic_encoder(hparams: dict = None) -> SemanticEncoder:
    global _semantic_encoder
    if _semantic_encoder is None:
        if hparams is None:
            hparams = get_hparams()
        name = hparams["ama-prof-divi"]["models"]["semantic"]["encoder"]["name"]
        if name == "mert":
            _semantic_encoder = MertEncoder_semantic.MertEncoder(hparams)
        else:
            raise ValueError(f"Unknown tokenizer: {name}")
    return _semantic_encoder