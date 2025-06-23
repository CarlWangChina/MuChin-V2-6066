from pathlib import Path
import unittest
from ama_prof_divi.utils import logging
from ama_prof_divi.configs import init_hparams, post_init_hparams, get_hparams
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.audioutils import get_audio_utils
from ama_prof_divi.models.semantic.encoding import get_semantic_encoder
from ama_prof_divi.models.semantic.chords_compressor import get_chords_compressor, get_chords_decompressor
logger = logging.getLogger(__name__)
TEST_AUDIO_FILE = "10035.mp3"

class TestSemanticEncoder(unittest.TestCase):
    def setUp(self):
        init_hparams()
        post_init_hparams()
        logger.info("Setup: building semantic encoder.")
        self.device = get_hparams()["ama-prof-divi"]["device"]
        self.encoder = get_semantic_encoder()
        self.chords_compressor = get_chords_compressor()
        self.chords_decompressor = get_chords_decompressor()
        self.au_utils = get_audio_utils()
        self.test_data_path = Path(__file__).parent.parent.joinpath('testdata')

    def test_build_semantic_encoder(self):
        logger.info('test_build_semantic_encoder')
        self.assertIsNotNone(self.encoder)
        self.assertIsNotNone(self.chords_compressor)
        self.assertIsNotNone(self.chords_decompressor)
        logger.info("encoder.model_name = %s" % self.encoder.model_name)

    def test_encoding(self):
        logger.info('test_encoding')
        audio, sampling_rate = self.au_utils.load_audio(self.test_data_path.joinpath(TEST_AUDIO_FILE))
        self.assertIsNotNone(audio)
        duration = 15
        audio = audio[:, :sampling_rate * duration]
        features = self.encoder(audio, sampling_rate)
        self.assertIsNotNone(features)
        self.assertEqual(features.dim(), 3)
        self.assertEqual(features.shape[0], 1)
        self.assertEqual(features.shape[2], self.encoder.features_dim)
        self.assertEqual(features.shape[1], duration * self.encoder.features_rate)
        logger.info(f"features.shape = {features.shape}")
        chords = self.chords_compressor(features.to(self.device))
        self.assertIsNotNone(chords)
        self.assertEqual(chords.dim(), 3)
        self.assertEqual(chords.shape[0], 1)
        self.assertEqual(chords.shape[2], self.chords_compressor.dim)
        self.assertEqual(chords.shape[1], features.shape[1] // self.chords_compressor.compress_ratio)
        logger.info(f"chords.shape = {chords.shape}")
        decompressed = self.chords_decompressor(chords)
        self.assertIsNotNone(decompressed)
        self.assertEqual(decompressed.shape[0], features.shape[0])
        self.assertEqual(decompressed.shape[1], chords.shape[1] * self.chords_compressor.compress_ratio)
        self.assertEqual(decompressed.shape[2], features.shape[2])