import unittest
import torch
from random import randint
from ama_prof_divi.utils import logging
from ama_prof_divi.configs import init_hparams, post_init_hparams
from ama_prof_divi.models.semantic.generators import get_melody_generator
from ama_prof_divi.modules.transformers import InferAccelerationCache
logger = logging.getLogger(__name__)

class TestMelodyGenerator(unittest.TestCase):
    def setUp(self):
        init_hparams()
        post_init_hparams()
        logger.info("Setup: building melody generator.")
        self.generator = get_melody_generator()

    def test_build_melody_generator(self):
        logger.info('test_build_melody_generator')
        self.assertIsNotNone(self.generator)
        logger.info("Melody generator: %s", self.generator)

    def test_generate_all_default(self):
        logger.info('test_generate_all_default')
        chords_seq = torch.randint(0, 100, (1, 50))
        cache = InferAccelerationCache(self.generator.model_args)
        self.generator.generate(chords_seq, max_gen_len=300, cache=cache)

    def test_forward_pass_for_training(self):
        logger.info("test_forward_pass_for_training")
        text_prompts = ["美妙的中文歌曲"]
        chords_seq = [[randint(0, 100) for _ in range(8)] for _ in range(len(text_prompts))]
        melody_seq = [[randint(0, 100) for _ in range(4 * 15)] for _ in range(len(text_prompts))]
        self.generator.forward_training(text_prompt=text_prompts, text_prompt_language="zh", chord_sequence=chords_seq, melody_sequence=melody_seq)