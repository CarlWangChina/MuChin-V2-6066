import unittest
import torch
from ama_prof_divi.utils import init_hparams, post_init_hparams
import logging
from ama_prof_divi.models.semantic.generators import get_chords_generator
from ama_prof_divi.modules.transformers import InferAccelerationCache
logger = logging.getLogger(__name__)

class TestChordsGenerator(unittest.TestCase):
    def setUp(self):
        init_hparams()
        post_init_hparams()
        logger.info("Setup: building chords generator.")
        self.generator = get_chords_generator()

    def test_build_chords_generator(self):
        logger.info('test_build_chords_generator')
        self.assertIsNotNone(self.generator)
        logger.info("Chords generator: %s", self.generator)

    def test_generate_all_default(self):
        logger.info('test_generate_all_default')
        cache = InferAccelerationCache(self.generator.model_args)
        chords = self.generator.generate(max_gen_len=30, cache=cache)
        self.assertIsNotNone(chords)
        self.assertEqual(len(chords), 1)
        self.assertLessEqual(len(chords[0]), 30)
        logger.info(f"Generated chords: {len(chords[0])} tokens.")

    def test_generate_with_text_prompt_only(self):
        logger.info("test_generate_with_text_prompt_only")
        text_prompts = ["美妙的中文歌曲", "摇滚，摇滚", "Rap rap rap"]
        cache = InferAccelerationCache(self.generator.model_args)
        chords = self.generator.generate(text_prompt=text_prompts, text_prompt_language="zh", max_gen_len=30, cache=cache)
        self.assertIsNotNone(chords)
        self.assertEqual(len(chords), len(text_prompts))
        for i in range(len(chords)):
            self.assertLessEqual(len(chords[i]), 30)

    def test_generate_with_chords_prompt_only(self):
        logger.info("test_generate_with_chords_prompt_only")
        chord_prompt = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 4, 5, 6], [7, 7, 9, 8, 0, 2, 1]]
        cache = InferAccelerationCache(self.generator.model_args)
        chords = self.generator.generate(chord_prompt=chord_prompt, cache=cache, max_gen_len=30)
        self.assertIsNotNone(chords)
        self.assertEqual(len(chords), len(chord_prompt))
        for i in range(len(chords)):
            self.assertLessEqual(len(chords[i]), 30)

    def test_generate_with_text_and_chords_prompt(self):
        logger.info("test_generate_with_text_and_chords_prompt")
        text_prompts = ["美妙的中文歌曲", "摇滚，摇滚", "Rap rap rap"]
        chord_prompt = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 4, 5, 6], [7, 7, 9, 8, 0, 2, 1]]
        cache = InferAccelerationCache(self.generator.model_args)
        chords = self.generator.generate(text_prompt=text_prompts, text_prompt_language="zh", chord_prompt=chord_prompt, cache=cache, max_gen_len=30)
        self.assertIsNotNone(chords)
        self.assertEqual(len(chords), len(text_prompts))
        for i in range(len(chords)):
            self.assertLessEqual(len(chords[i]), 30)

    def test_forward_pass(self):
        logger.info("test_forward_pass")
        sentence = torch.randint(0, self.generator.vocab_size, (30,))
        sentence[0] = self.generator.tokenizer.start_id
        prompt_size = 1
        all_tokens = sentence.expand(sentence.shape[0], -1)
        all_tokens = (torch.tril(all_tokens, diagonal=prompt_size - 1) + torch.triu(torch.ones_like(all_tokens) * self.generator.tokenizer.pad_id, diagonal=prompt_size)).to(self.generator.device)
        context = torch.randn((30, 10, self.generator.dim)).to(self.generator.device)
        labels = torch.full(all_tokens.shape, self.generator.tokenizer.pad_id, dtype=torch.long)
        for i in range(all_tokens.shape[0] - 1):
            labels[i, i + prompt_size - 1] = all_tokens[i + 1, i + prompt_size]
        labels[-1, -1] = self.generator.tokenizer.end_id
        labels = labels.to(self.generator.device)
        result = self.generator(all_tokens, context=context, labels=labels)
        self.assertIsNotNone(result)
        self.assertEqual(result["logits"].shape, (30, 30, self.generator.vocab_size))
        self.assertEqual(result["loss"].shape, ())
        logger.info("logits: {}".format(result["logits"].shape))
        logger.info("loss: {}".format(result["loss"]))

    def test_forward_pass_for_training(self):
        logger.info("test_forward_pass_for_training")
        text_prompts = ["美妙的中文歌曲", "摇滚，摇滚", "Rap rap rap"]
        lyrics = ["<|ss_verse|>冰封的湖面，银装素裹的城，静谧的晨曦，霜花在窗上凝。", "<|ss_verse|>咚咚锵 咚咚锵", "<|ss_verse|>Wo ka ka ka"]
        chord_seq = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 4, 5, 6], [7, 7, 9, 8, 0, 2, 1]]
        result = self.generator.forward_training(text_prompt=text_prompts, text_prompt_language="zh", lyrics=lyrics, chord_sequences=chord_seq)
        self.assertIsNotNone(result)
        self.assertEqual(result["logits"].shape[2], self.generator.vocab_size)
        self.assertEqual(result["loss"].shape, ())
        logger.info("logits: {}".format(result["logits"].shape))
        logger.info("loss: {}".format(result["loss"]))