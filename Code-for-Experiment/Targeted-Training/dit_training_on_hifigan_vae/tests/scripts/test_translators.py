import unittest
from music_dit.modules.translators import TranslatorZhToEn, TranslatorEnToZh
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger

logger = get_logger(__name__)

class TestTranslators(unittest.TestCase):
    def setUp(self):
        logger.info("Setting up translators.")
        self.translator_zh_to_en = TranslatorZhToEn()
        self.translator_en_to_zh = TranslatorEnToZh()

    def test_translate_zh_to_en(self):
        logger.info("Testing translation from Chinese to English.")
        text = "你好，世界！"
        translated_text = self.translator_zh_to_en.translate(text)
        self.assertIsNotNone(translated_text)
        logger.info(f"Translated text: {translated_text}")

    def test_translate_en_to_zh(self):
        logger.info("Testing translation from English to Chinese.")
        text = "Hello, world!"
        translated_text = self.translator_en_to_zh.translate(text)
        self.assertIsNotNone(translated_text)
        logger.info(f"Translated text: {translated_text}")