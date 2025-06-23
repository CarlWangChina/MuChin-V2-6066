import unittest
from music_dit.utils import get_hparams, get_logger

logger = get_logger(__name__)

class TestHyperParameters(unittest.TestCase):
    def test_get_hparams(self):
        logger.info("Testing get_hparams()")
        hparams = get_hparams()
        self.assertIsNotNone(hparams)
        logger.info("hparams: %s", hparams)