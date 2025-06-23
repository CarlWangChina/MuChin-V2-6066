from ama_prof_divi.configs import get_hparams
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.modules.fastspeech2.predictors.duration_predictor import DurationPredictor
from .chords_generator import ChordsGenerator
from .melody_generator import MelodyGenerator

_duration_predictor = None
_chords_generator = None
_melody_generator = None

def get_chords_generator(hparams: dict = None) -> ChordsGenerator:
    global _chords_generator
    if _chords_generator is None:
        if hparams is None:
            hparams = get_hparams()
        _chords_generator = ChordsGenerator(hparams)
    return _chords_generator

def get_melody_generator(hparams: dict = None) -> MelodyGenerator:
    global _melody_generator
    if _melody_generator is None:
        if hparams is None:
            hparams = get_hparams()
        _melody_generator = MelodyGenerator(hparams)
    return _melody_generator

def get_duration_predictor(hparams: dict = None) -> DurationPredictor:
    global _duration_predictor
    if _duration_predictor is None:
        if hparams is None:
            hparams = get_hparams()
        _duration_predictor = DurationPredictor(hparams)
    return _duration_predictor