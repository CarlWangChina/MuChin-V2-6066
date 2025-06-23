import unittest
from ama_prof_divi.utils import logging
from ama_prof_divi.configs import init_hparams, post_init_hparams
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.models.lyrics.builder import get_lyrics_tokenizer
_tokenizerlogger = logging.getLogger(__name__)
class TestTokenizer(unittest.TestCase):
    def setUp(self):
        init_hparams()
        post_init_hparams()
        _tokenizerlogger.info("Setup: building tokenizer.")
        self.tokenizer = get_lyrics_tokenizer()
    def test_build_tokenizer(self):
        _tokenizerlogger.info('test_build_tokenizer')
        self.assertIsNotNone(self.tokenizer)
        self.assertIsNotNone(self.tokenizer.model_name)
        _tokenizerlogger.info("tokenizer.model_name = %s" % self.tokenizer.model_name)
        self.assertGreater(self.tokenizer.vocab_size, 0)
        _tokenizerlogger.info("tokenizer.vocab_size = %d" % self.tokenizer.vocab_size)
        _tokenizerlogger.info("tokenizer = %s" % self.tokenizer)
    def test_encode_decode(self):
        _tokenizerlogger.info("test_encode_decode")
        text = "Hello, world!"
        tokens_by_forward = self.tokenizer(text)
        tokens_by_encode = self.tokenizer.encode(text)
        self.assertEqual(tokens_by_forward, tokens_by_encode)
        self.assertGreaterEqual(len(tokens_by_forward), 2)
        decoded = self.tokenizer.decode(tokens_by_forward)
        self.assertEqual(decoded, text)
    def test_encode_decode_special(self):
        _tokenizerlogger.info("test_encode_decode_special")
        special_token_strings = self.tokenizer.special_tokens_set()
        self.assertGreater(len(special_token_strings), 0)
        for token_str in special_token_strings:
            token = self.tokenizer.encode(token_str)
            self.assertEqual(len(token), 1)
            decoded = self.tokenizer.decode(token)
            self.assertEqual(decoded, token_str)
    def test_encode_decode_batch(self):
        _tokenizerlogger.info("test_encode_decode_batch")
        texts = ["Hello, world!", "<|ss_start|> Hello, world! <|ss_end|>", "<|ss_start|> Hello, world! <|ss_end|> <|ss_start|> Hello, world! <|ss_end|>"]
        tokens = self.tokenizer.encode_batch(texts)
        self.assertEqual(len(tokens), len(texts))
        for i in range(len(texts)):
            token = self.tokenizer(texts[i])
            self.assertEqual(token, tokens[i])
        decoded = self.tokenizer.decode_batch(tokens)
        self.assertEqual(len(decoded), len(texts))
        for i in range(len(texts)):
            self.assertEqual(decoded[i], texts[i])
    def test_check_special_tokens(self):
        _tokenizerlogger.info('test_check_special_tokens')
        token = "<|ss_start|>"
        self.assertTrue(self.tokenizer.is_special_token(token))
        ids = self.tokenizer.encode(token)
        self.assertEqual(len(ids), 1)
        self.assertTrue(self.tokenizer.is_special_token(ids[0]))
        token = "H"
        self.assertFalse(self.tokenizer.is_special_token(token))
        ids = self.tokenizer.encode(token)
        self.assertFalse(self.tokenizer.is_special_token(ids[0]))