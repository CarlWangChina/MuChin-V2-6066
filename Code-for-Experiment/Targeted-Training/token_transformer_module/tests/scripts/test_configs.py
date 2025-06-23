import unittest
from ama_prof_divi.configs import *
from ama_prof_divi.utils import logging

logger = logging.getLogger(__name__)

class TestLogger(unittest.TestCase):
    def test_logger(self):
        logger.info('test_logger')
        self.assertIsNotNone(logger)

class TestHparams(unittest.TestCase):
    def setUp(self):
        init_hparams()
        post_init_hparams()

    def test_hparams(self):
        logger.info('test_hparams')
        hparams = get_hparams()
        logger.info(f"hparams = {hparams}")
        self.assertIsNotNone(hparams)
        self.assertGreater(len(hparams), 0)

if __name__ == '__main__':
    unittest.main()