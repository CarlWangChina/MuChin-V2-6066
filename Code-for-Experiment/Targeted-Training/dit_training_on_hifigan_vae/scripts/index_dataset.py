import torch
from music_dit.utils import get_hparams, get_logger
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.dm.data_processing import make_data_index

hparams = get_hparams()
logger = get_logger(__name__)

source_audio_dir = hparams.data.original.audio_dir
source_lyrics_dir = hparams.data.original.lyrics_dir
target_index_dir = hparams.data.index.index_dir

logger.info("Source audio dir: %s", source_audio_dir)
logger.info("Source lyrics dir: %s", source_lyrics_dir)
logger.info("Target index dir: %s", target_index_dir)

audio_suffix = hparams.data.original.audio_suffix
lyrics_suffix = hparams.data.original.lyrics_suffix
approx_audio_file_num = hparams.data.original.approx_audio_file_num
approx_lyrics_file_num = hparams.data.original.approx_lyrics_file_num

if __name__ == "__main__":
    make_data_index(source_audio_dir, source_lyrics_dir, target_index_dir,
                     audio_suffix=audio_suffix, lyrics_suffix=lyrics_suffix,
                     approx_audio_file_num=approx_audio_file_num,
                     approx_lyrics_file_num=approx_lyrics_file_num)