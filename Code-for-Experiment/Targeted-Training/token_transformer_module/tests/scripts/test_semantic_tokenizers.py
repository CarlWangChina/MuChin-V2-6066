import torch
import unittest
import torch.nn.functional as F
from ama_prof_divi.utils import logging
from ama_prof_divi.configs import init_hparams, post_init_hparams
from ama_prof_divi.models.semantic.tokenizers import *

logger = logging.getLogger(__name__)

class TestSemanticTokenizers(unittest.TestCase):
    def setUp(self):
        init_hparams()
        post_init_hparams()

    def test_build_melody_tokenizer(self):
        logger.info('test_build_melody_tokenizer')
        tokenizer = build_melody_tokenizer_unfitted()
        self.assertIsNotNone(tokenizer)
        self.assertFalse(tokenizer.is_fitted)

    def test_get_singletons(self):
        logger.info('test_get_singletons')
        m_tokenizer = get_melody_tokenizer()
        self.assertIsNotNone(m_tokenizer)
        self.assertTrue(m_tokenizer.is_fitted)
        t = get_melody_tokenizer()
        self.assertEqual(m_tokenizer, t)

    def test_tokenize(self):
        logger.info('test_tokenize')
        m_tokenizer = get_melody_tokenizer()
        self.assertIsNotNone(m_tokenizer)
        self.assertTrue(m_tokenizer.is_fitted)
        melody = torch.randn(3, 102, 1024)
        tokens = m_tokenizer.tokenize(melody)
        logger.info('tokens.shape = %s' % str(tokens.shape))
        self.assertEqual(tokens.shape, (melody.shape[0], melody.shape[1], m_tokenizer.num_quantizers))
        y = m_tokenizer.decode(tokens)
        self.assertEqual(y.shape, melody.shape)
        loss = F.mse_loss(y, melody)
        logger.info('loss_full = %f' % loss.item())
        tokens = m_tokenizer.tokenize(melody, num_q=6)
        self.assertEqual(tokens.shape, (melody.shape[0], melody.shape[1], 6))
        self.assertEqual(tokens.dtype, torch.long)
        y = m_tokenizer.decode(tokens)
        self.assertEqual(y.shape, melody.shape)
        loss = F.mse_loss(y, melody)
        logger.info('loss_partial_levels = %f' % loss.item())