import unittest
from pathlib import Path
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.modules.clap.clap import ClapEncoder
from music_dit.utils import get_logger, get_audio_utils

logger = get_logger(__name__)
TEST_AUDIO_FILE = "654a3a9be052b2c2cb074230b390016697ad7f40_src.mp3"

class TestClapEncoder(unittest.TestCase):
    def setUp(self):
        logger.info("Setting up the Clap encoder.")
        self.clap_encoder = ClapEncoder()
        self.au_utils = get_audio_utils()
        self.test_data_path = Path(__file__).parent.parent.joinpath('testdata')

    def test_clap_encoder(self):
        logger.info("Testing the Clap encoder.")
        self.assertIsNotNone(self.clap_encoder)
        logger.info("Sampling rate: %d", self.clap_encoder.sampling_rate)
        logger.info("Number of channels: %d", self.clap_encoder.num_channels)
        logger.info("Joint embedding dimension: %d", self.clap_encoder.joint_embedding_dim)

    def test_text_embedding(self):
        logger.info("Testing text embedding.")
        texts = ["A pop song with a happy mood."]
        embedding = self.clap_encoder.get_text_embedding(texts)
        self.assertIsNotNone(embedding)
        self.assertEqual(embedding.shape, (len(texts), self.clap_encoder.joint_embedding_dim))
        texts = ["Hello world!", "A bird is singing in the tree"]
        embedding = self.clap_encoder.get_text_embedding(texts)
        self.assertIsNotNone(embedding)
        self.assertEqual(embedding.shape, (len(texts), self.clap_encoder.joint_embedding_dim))
        texts_zh = ["一首好听的流行歌曲"]
        embedding = self.clap_encoder.get_text_embedding_zh(texts_zh)
        self.assertIsNotNone(embedding)
        self.assertEqual(embedding.shape, (len(texts_zh), self.clap_encoder.joint_embedding_dim))
        texts_zh = ["你好世界！", "一只鸟在树上唱歌"]
        embedding = self.clap_encoder.get_text_embedding_zh(texts_zh)
        self.assertIsNotNone(embedding)
        self.assertEqual(embedding.shape, (len(texts_zh), self.clap_encoder.joint_embedding_dim))

    def test_audio_embedding(self):
        logger.info('test_audio_embedding')
        self.assertEqual(self.clap_encoder.sampling_rate, 48000)
        audio, _ = self.au_utils.load_audio(self.test_data_path.joinpath(TEST_AUDIO_FILE))
        audio = audio[:1, :48000]
        embedding = self.clap_encoder.get_audio_embedding(audio)
        self.assertIsNotNone(embedding)
        self.assertEqual(embedding.shape, (1, self.clap_encoder.joint_embedding_dim))