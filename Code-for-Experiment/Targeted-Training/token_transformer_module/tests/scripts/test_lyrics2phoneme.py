import unittest
from ama_prof_divi.utils import logging
from ama_prof_divi.configs import init_hparams, post_init_hparams
from ama_prof_divi.models.lyrics import get_lyrics_to_phoneme_translator
_translatorlogger = logging.getLogger(__name__)

class Lyrics2PhonemeTest(unittest.TestCase):
    def setUp(self):
        init_hparams()
        post_init_hparams()
        _translatorlogger.info("Setup: building lyrics-to-phoneme translator.")
        self.translator = get_lyrics_to_phoneme_translator()

    def test_build_translator(self):
        self.assertIsNotNone(self.translator)
        _translatorlogger.info("Lyrics-to-phoneme translator: {}".format(self.translator))

    def test_split_lyrics(self):
        lyrics = ("<|ss_verse|> I'm a test case\'. <|ss_chorus|> "
                  "Hello, Hello hello 你好吗？\n我很好！")
        sentences = self.translator.split_sentence(lyrics)
        _translatorlogger.info("Lyrics: {}".format(lyrics))
        _translatorlogger.info("Sentences: {}".format(sentences))
        expected = [('<|ss_verse|>', 'special_tokens'),
                    ('I\'m a test case\'', 'english'),
                    ('<|ss_sep|>', 'sep'),
                    ('<|ss_chorus|>', 'special_tokens'),
                    ('Hello', 'english'),
                    ('<|ss_sep|>', 'sep'),
                    ('Hello hello', 'english'),
                    ('你好吗', 'chinese'),
                    ('<|ss_sep|>', 'sep'),
                    ('我很好', 'chinese'),
                    ('<|ss_sep|>', 'sep')]

        self.assertEqual(expected, sentences)
        phonemes = self.translator.split_and_translate_sentence(lyrics)
        _translatorlogger.info(phonemes)

    def test_encode_lyrics(self):
        lyrics = ["<|ss_verse|> I'm a test case. <|ss_chorus|>",
                  "Hello, Hello hello 你好吗？\n我很好！"]

        tokens = self.translator(lyrics)
        self.assertEqual(tokens.dim(), 2)
        self.assertEqual(tokens.shape[0], 2)
        _translatorlogger.info("tokens: %s", tokens)