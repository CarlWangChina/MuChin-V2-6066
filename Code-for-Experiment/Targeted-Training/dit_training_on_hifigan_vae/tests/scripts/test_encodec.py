import unittest
import torch
import torchaudio
from pathlib import Path
from music_dit.utils import get_logger, get_audio_utils, get_hparams
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.modules.encodec.encodec import EncodecVAE

logger = get_logger(__name__)
TEST_AUDIO_FILE = "654a3a9be052b2c2cb074230b390016697ad7f40_src.mp3"

class TestEncodec(unittest.TestCase):
    def setUp(self):
        self.au_utils = get_audio_utils()
        self.encodec_vae = EncodecVAE()
        self.test_data_path = Path(__file__).parent.parent.joinpath('testdata')

    def test_build_encodec_wrapper(self):
        logger.info("test_build_encodec_wrapper")
        self.assertIsNotNone(self.encodec_vae)
        logger.info("num_channels = %d", self.encodec_vae.num_channels)
        logger.info("sampling_rate = %d", self.encodec_vae.sampling_rate)
        logger.info("segment_length = %d", self.encodec_vae.segment_length)
        logger.info("segment_stride = %d", self.encodec_vae.segment_stride)
        logger.info("frame_rate = %d", self.encodec_vae.frame_rate)

    def test_encodec(self):
        logger.info("test_encodec")
        audio, sampling_rate = torchaudio.load(self.test_data_path.joinpath(TEST_AUDIO_FILE))
        self.assertIsNotNone(audio)
        logger.info("Audio shape before encoding: %s, sampling rate: %d", audio.shape, sampling_rate)
        with torch.no_grad():
            vae_encoded, scales = self.encodec_vae.encode(audio, sampling_rate)
            self.assertIsNotNone(vae_encoded)
            self.assertEqual(vae_encoded.shape[0], scales.shape[0])
            self.assertEqual(vae_encoded.shape[-1], self.encodec_vae.embedding_dim)
            logger.info("vae_encoded.shape = %s", vae_encoded.shape)
            logger.info("scales.shape = %s", scales.shape)
            logger.info("Decoding the encoded data...")
            audio = self.encodec_vae.decode(vae_encoded, scales)
            self.assertIsNotNone(audio)
            logger.info("Audio shape after decoding: %s, sampling rate: %d", audio.shape, self.encodec_vae.sampling_rate)
            output_file = get_hparams().data_dir.joinpath("vae_output.wav")
            self.au_utils.save_audio(audio, self.encodec_vae.sampling_rate, output_file)
            logger.info("Saved output_file as %s", output_file)