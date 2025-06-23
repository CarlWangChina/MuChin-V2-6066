import unittest
from pathlib import Path
from ama_prof_divi.utils import logging
from ama_prof_divi.configs import init_hparams, post_init_hparams
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.audioutils import get_audio_utils
import get_audio_utils
import get_prompt_encoder
logger = logging.getLogger(__name__)
TEST_AUDIO_FILE = "10035.mp3"

class TestPromptEncoder(unittest.TestCase):
    def setUp(self):
        init_hparams()
        post_init_hparams()
        logger.info("Setup: building prompt encoder.")
        self.au_utils = get_audio_utils()
        self.encoder = get_prompt_encoder()
        self.test_data_path = Path(__file__).parent.parent.joinpath('testdata')

    def test_build_prompt_encoder(self):
        logger.info('test_build_prompt_encoder')
        self.assertIsNotNone(self.encoder)
        logger.info(f"Model name: {self.encoder.model_name}")
        logger.info(f"Checkpoint: {self.encoder.checkpoint}")
        self.assertGreater(self.encoder.sampling_rate, 0)
        self.assertGreater(self.encoder.max_clip_samples, 0)
        self.assertGreater(self.encoder.vocab_size, 0)
        self.assertGreater(self.encoder.joint_embedding_dim, 0)
        logger.info("Prompt encoder: {}".format(self.encoder))

    def test_text_embedding(self):
        logger.info('test_text_embedding')
        texts = ["A pop song with a happy mood."]
        embedding = self.encoder.get_text_embedding(texts)
        self.assertIsNotNone(embedding)
        self.assertEqual(embedding.shape, (len(texts), self.encoder.joint_embedding_dim))
        texts = ["Hello world!", "A bird is singing in the tree"]
        embedding = self.encoder.get_text_embedding(texts)
        self.assertIsNotNone(embedding)
        self.assertEqual(embedding.shape, (len(texts), self.encoder.joint_embedding_dim))
        texts_zh = ["一首好听的流行歌曲"]
        embedding = self.encoder.get_text_embedding_zh(texts_zh)
        self.assertIsNotNone(embedding)
        self.assertEqual(embedding.shape, (len(texts_zh), self.encoder.joint_embedding_dim))
        texts_zh = ["你好世界！", "一只鸟在树上唱歌"]
        embedding = self.encoder.get_text_embedding_zh(texts_zh)
        self.assertIsNotNone(embedding)
        self.assertEqual(embedding.shape, (len(texts_zh), self.encoder.joint_embedding_dim))

    def test_text_embedding_zh_cn(self):
        logger.info('test_text_embedding_zh_cn')

    def test_audio_embedding(self):
        logger.info('test_audio_embedding')
        self.assertEqual(self.encoder.sampling_rate, 48000)
        audio, _ = self.au_utils.load_audio(self.test_data_path.joinpath(TEST_AUDIO_FILE))
        audio = audio[:1, :self.encoder.max_clip_samples]
        embedding = self.encoder.get_audio_embedding(audio)
        self.assertIsNotNone(embedding)
        self.assertEqual(embedding.shape, (1, self.encoder.joint_embedding_dim))