import unittest
from pathlib import Path
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.logging import get_logger
from Code_for_Experiment.Targeted_Training.text_to_phoneme_converter.ama-prof-divi_text.phoneme.phoneme import Lyrics2Phoneme

logger = get_logger(__name__)

class TestLyrics2Phoneme(unittest.TestCase):
    def setUp(self):
        self.lyrics2phoneme = Lyrics2Phoneme()
        logger.info("Lyrics2Phoneme instance is created.")
        logger.info("Loading test data...")
        test_data_file = Path(__file__).parent.parent / "testdata" / "lyrics.txt"
        self.test_data = []
        with open(test_data_file, "r") as f:
            for line in f:
                line = line.strip()
                if line != "":
                    self.test_data.append(line)
        logger.info("Test data loaded. %d lyrics.", len(self.test_data))

    def test_model(self):
        self.assertIsNotNone(self.lyrics2phoneme)

    def test_split_sentences(self):
        for i, lyrics in enumerate(self.test_data):
            sentences = self.lyrics2phoneme.split_sentence(lyrics)
            self.assertGreater(len(sentences), 0)

    def test_lyrics_to_phoneme(self):
        for i, lyrics in enumerate(self.test_data):
            phoneme = self.lyrics2phoneme.translate(lyrics)
            logger.info("%d: %s", i, phoneme)