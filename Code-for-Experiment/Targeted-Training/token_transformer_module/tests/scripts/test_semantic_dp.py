import unittest
import torch
from ama_prof_divi.utils import init_hparams, post_init_hparams
import logging
from ama_prof_divi.models.semantic.generators import get_duration_predictor
from ama_prof_divi.models.lyrics import get_phoneme_tokenizer
from ama_prof_divi.models.song_structures import get_ss_tokenizer
logger = logging.getLogger(__name__)
class TestDurationPredictor(unittest.TestCase):
    def setUp(self):
        init_hparams()
        post_init_hparams()
        self.dp = get_duration_predictor()
        self.phoneme_tokenizer = get_phoneme_tokenizer()
        self.ss_tokenizer = get_ss_tokenizer()
        self.device = self.dp.device
    def test_get_dp(self):
        logger.info("Testing get_duration_predictor()...")
        self.assertIsNotNone(self.dp)
    def get_context(self, seq_len: int = 100):
        phoneme_seq = torch.randint(0, self.phoneme_tokenizer.vocab_size, (2, seq_len)).to(self.device)
        ss_seq = torch.randint(0, self.ss_tokenizer.vocab_size, (2, seq_len)).to(self.device)
        return self.dp.encode_context(phonemes=phoneme_seq, ss=ss_seq)
    def test_dp_forward(self):
        logger.info("Testing duration predictor forward()...")
        context = self.get_context()
        self.assertEqual(context.shape, (2, 100, self.dp.dim))
        dur = torch.randint(0, 100, (2, 200)).to(self.device)
        ss = torch.randint(0, self.ss_tokenizer.vocab_size, (2, 200)).to(self.device)
        pause = torch.randint(0, 2, (2, 200)).to(self.device)
        output = self.dp(duration=dur, ss=ss, pause=pause, context=context)
        self.assertEqual(output["duration_logits"].shape, (2, 200))
        self.assertEqual(output["pause_logits"].shape, (2, 200, self.dp.pause_vocab_size))
        self.assertEqual(output["ss_logits"].shape, (2, 200, self.dp.ss_vocab_size))
    def test_dp_training(self):
        logger.info("Testing duration predictor training process...")
        context = self.get_context(seq_len=10)
        self.assertEqual(context.shape, (2, 10, self.dp.dim))
        dur = torch.randint(0, 100, (2, 20)).to(self.device)
        ss = torch.randint(0, self.ss_tokenizer.vocab_size, (2, 20)).to(self.device)
        pause = torch.randint(0, 2, (2, 20)).to(self.device)
        output = self.dp.perform_training(duration=dur, ss=ss, pause=pause, context=context)
        self.assertEqual(output["loss"].shape, ())
    def test_dp_generator(self):
        logger.info("Testing duration predictor generator...")
        context = self.get_context()
        self.assertEqual(context.shape, (2, 100, self.dp.dim))
        output = self.dp.generate(context=context, max_gen_len=200)
        self.assertEqual(output["duration"].shape, (2, 201))
        self.assertEqual(output["pause"].shape, (2, 201))
        self.assertEqual(output["ss"].shape, (2, 201))