import unittest
from pathlib import Path
from music_dit.utils import get_logger, Lyrics, LyricFileParser

logger = get_logger(__name__)
TEST_LYRIC_FILE = "654a3a9be052b2c2cb074230b390016697ad7f40_src.txt"

class TestLyricFileParser(unittest.TestCase):
    def setUp(self):
        self.lyric_parser = LyricFileParser()
        self.test_data_path = Path(__file__).parent.parent.joinpath('testdata')

    def test_lyric_parser(self):
        logger.info("Testing the lyric parser.")
        lyrics = self.lyric_parser.parse(self.test_data_path / TEST_LYRIC_FILE)
        self.assertIsNotNone(lyrics)
        if len(lyrics) > 0:
            cat_lyrics = lyrics[0].cat(*lyrics[1:])
            logger.info("Concatenated lyrics: %s", cat_lyrics)