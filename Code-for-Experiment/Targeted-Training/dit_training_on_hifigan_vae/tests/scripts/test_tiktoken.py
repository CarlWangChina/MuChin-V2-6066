import unittest
from music_dit.utils import get_logger
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.modules.tiktoken.tiktoken import TikTokenWrapper

logger = get_logger(__name__)

class TestTiktoken(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TikTokenWrapper()

    def test_tiktoken(self):
        logger.info("Testing TikToken wrapper.")
        logger.info("Vocabulary size: %d" % self.tokenizer.encoding.n_vocab)
        logger.info("Base vocabulary size: %d" % self.tokenizer.base_encoding.n_vocab)
        logger.info("Special tokens: %s" % self.tokenizer.special_tokens_set)

    def test_encode_decode(self):
        logger.info("test_encode_decode")
        text = "Hello, world!"
        tokens_by_forward = self.tokenizer(text)
        tokens_by_encode = self.tokenizer.encode(text)
        self.assertEqual(tokens_by_forward, tokens_by_encode)
        self.assertGreaterEqual(len(tokens_by_forward), 2)
        decoded = self.tokenizer.decode(tokens_by_forward)
        self.assertEqual(decoded, text)

    def test_encode_decode_special(self):
        logger.info("test_encode_decode_special")
        special_token_strings = self.tokenizer.special_tokens_set
        self.assertGreater(len(special_token_strings), 0)
        for token_str in special_token_strings:
            token = self.tokenizer.encode(token_str)
            self.assertEqual(len(token), 1)
            decoded = self.tokenizer.decode(token)
            self.assertEqual(decoded, token_str)

    def test_encode_decode_batch(self):
        logger.info("test_encode_decode_batch")
        texts = ["Hello, world!",
                 "<|ss_start|> Hello, world! <|ss_end|>",
                 "<|ss_start|> Hello, world! <|ss_end|> <|ss_start|> Hello, world! <|ss_end|>"]
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
        logger.info('test_check_special_tokens')
        token = "<|ss_start|>"
        self.assertTrue(self.tokenizer.is_special_token(token))
        ids = self.tokenizer.encode(token)
        self.assertEqual(len(ids), 1)
        self.assertTrue(self.tokenizer.is_special_token(ids[0]))
        token = "H"
        self.assertFalse(self.tokenizer.is_special_token(token))
        ids = self.tokenizer.encode(token)
        self.assertFalse(self.tokenizer.is_special_token(ids[0]))