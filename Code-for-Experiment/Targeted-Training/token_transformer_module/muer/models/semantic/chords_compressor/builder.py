from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.hparams import get_hparams
from .chords_compressor import ChordsCompressor
from .chords_decompressor import ChordsDecompressor
from .baseline import ChordsCompressorBaseline, ChordsDecompressorBaseline

_chords_compressor = None
_chords_compressor_baseline = None
_chords_decompressor = None
_chords_decompressor_baseline = None

def get_chords_compressor(hparams: dict = None) -> ChordsCompressor:
    global _chords_compressor
    if _chords_compressor is None:
        if hparams is None:
            hparams = get_hparams()
        _chords_compressor = ChordsCompressor(hparams)
    return _chords_compressor

def get_chords_compressor_baseline(hparams: dict = None) -> ChordsCompressorBaseline:
    global _chords_compressor_baseline
    if _chords_compressor_baseline is None:
        if hparams is None:
            hparams = get_hparams()
        _chords_compressor_baseline = ChordsCompressorBaseline(hparams)
    return _chords_compressor_baseline

def get_chords_decompressor(hparams: dict = None) -> ChordsDecompressor:
    global _chords_decompressor
    if _chords_decompressor is None:
        if hparams is None:
            hparams = get_hparams()
        _chords_decompressor = ChordsDecompressor(hparams)
    return _chords_decompressor

def get_chords_decompressor_baseline(hparams: dict = None) -> ChordsDecompressorBaseline:
    global _chords_decompressor_baseline
    if _chords_decompressor_baseline is None:
        if hparams is None:
            hparams = get_hparams()
        _chords_decompressor_baseline = ChordsDecompressorBaseline(hparams)
    return _chords_decompressor_baseline