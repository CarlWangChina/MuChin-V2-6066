import torch
import unittest
from ama_prof_divi.utils import logging
from ama_prof_divi.configs import init_hparams, post_init_hparams
from ama_prof_divi.models.lyrics import SongStructureEncoder

logger = logging.getLogger(__name__)
TEST_LYRICS = []

class SongStructureEncoderTest(unittest.TestCase):
    def setUp(self):
        init_hparams()
        post_init_hparams()
        logger.info("Setup: building song structure encoder.")
        self.encoder = SongStructureEncoder()

    def test_build_encoder(self):
        logger.info("test_build_encoder")
        self.assertIsNotNone(self.encoder)

    def test_encode_lyrics(self):
        logger.info("test_encode_lyrics")
        lyrics = TEST_LYRICS
        encoded_lyrics = self.encoder(lyrics)
        logger.info(f"Encoded lyrics: {encoded_lyrics.shape}.")
        self.assertEqual(encoded_lyrics.dim(), 3)
        self.assertEqual(encoded_lyrics.size(0), 2)
        self.assertEqual(encoded_lyrics.size(2), self.encoder.dim)