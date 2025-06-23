import unittest
import torch
import torchaudio
from pathlib import Path
from music_dit.utils import get_logger, get_hparams, LyricFileParser
from music_dit.models import MusicDiTPreprocessor, MusicDiTModel

logger = get_logger(__name__)
TEST_AUDIO_FILE = "654a3a9be052b2c2cb074230b390016697ad7f40_src.mp3"
TEST_LYRIC_FILE = "654a3a9be052b2c2cb074230b390016697ad7f40_src.txt"

class TestMusicDiTModel(unittest.TestCase):
    def setUp(self):
        self.hparams = get_hparams()
        self.device = self.hparams.device
        self.preprocessor = MusicDiTPreprocessor()
        self.music_dit_model = MusicDiTModel(num_layers=2, num_heads=4)
        self.lyric_parser = LyricFileParser()
        self.test_data_path = Path(__file__).parent.parent.joinpath('testdata')
        self.test_audio, self.original_sampling_rate = torchaudio.load(self.test_data_path.joinpath(TEST_AUDIO_FILE))
        self.vae_frame_size = self.preprocessor.vae_frame_size
        self.input_dim = self.preprocessor.input_dim
        self.hidden_dim = self.preprocessor.hidden_dim
        self.clap_dim = self.preprocessor.clap_embedding_dim
        self.lyric_vocab_size = self.preprocessor.lyric_vocab_size

    def test_preprocessor(self):
        logger.info("Testing the preprocessor.")
        self.assertIsNotNone(self.preprocessor)
        max_batches = 16
        min_length = 20
        max_length = 30
        prompt_length = 20
        split_data = self.preprocessor.split_audio(self.test_audio,
                                                   sampling_rate=self.original_sampling_rate,
                                                   max_batches=max_batches,
                                                   min_length=min_length,
                                                   max_length=max_length,
                                                   prompt_length=prompt_length)
        self.assertEqual(len(split_data), max_batches)
        for i in range(len(split_data)):
            split_data_dict = split_data[i]
            self.assertEqual(split_data_dict['audio'].size(),
                             (self.test_audio.size(0), split_data_dict['sampling_rate'] * max_length))
            self.assertEqual(split_data_dict['sampling_rate'], self.original_sampling_rate)
            self.assertEqual(split_data_dict['vae'].size(), (max_length, self.vae_frame_size, self.input_dim))
            self.assertEqual(split_data_dict['vae_scales'].size(), (max_length,))
            self.assertEqual(split_data_dict['prompt'].size(), (prompt_length,
                                                                self.vae_frame_size, self.input_dim))
            self.assertEqual(split_data_dict['clap'].size(), (max_length, self.clap_dim))
            self.assertIsNone(split_data_dict.get('lyrics'))
        lyrics = self.lyric_parser.parse(self.test_data_path / TEST_LYRIC_FILE)
        split_data = self.preprocessor.split_audio(self.test_audio,
                                                   sampling_rate=self.original_sampling_rate,
                                                   max_batches=10,
                                                   lyrics=lyrics)
        for i in range(len(split_data)):
            split_data_dict = split_data[i]
            self.assertGreaterEqual(split_data_dict['audio'].size(1),
                                    int(split_data_dict['sampling_rate'] * min_length))
            self.assertLessEqual(split_data_dict['audio'].size(1),
                                 int(split_data_dict['sampling_rate'] * max_length))
            self.assertEqual(split_data_dict['sampling_rate'], self.original_sampling_rate)
            self.assertEqual(split_data_dict['vae'].shape[1:3], (self.vae_frame_size, self.input_dim))
            self.assertEqual(split_data_dict['vae_scales'].size(0), split_data_dict['vae'].size(0))
            self.assertGreaterEqual(split_data_dict['vae'].size(0), min_length)
            self.assertLessEqual(split_data_dict['vae'].size(0), max_length)
            self.assertEqual(split_data_dict['prompt'].size(), (prompt_length,
                                                                self.vae_frame_size, self.input_dim))
            self.assertEqual(split_data_dict['clap'].size(1), self.clap_dim)
            if split_data_dict['lyrics'] is not None:
                self.assertEqual(split_data_dict['lyrics'].dim(), 1)

    def test_training_step(self):
        logger.info("Testing the training step.")
        vae_samples = torch.randn(4, 30, self.vae_frame_size, self.input_dim).to(self.device)
        clap = torch.randn(4, 30, self.clap_dim).to(self.device)
        prompt = torch.randn(4, 20, self.vae_frame_size, self.input_dim).to(self.device)
        lyrics = torch.randint(0, self.lyric_vocab_size, (4, 18)).to(self.device)
        padding_mask = torch.randint(0, 2, (4, 30)).to(self.device)
        with torch.no_grad():
            inp = self.music_dit_model.preprocess_input(x=vae_samples,
                                                        clap=clap,
                                                        prompt=prompt,
                                                        lyrics=lyrics,
                                                        padding_mask=padding_mask)
            loss = self.music_dit_model.training_step(inp)
        self.assertIsNotNone(loss)
        self.assertEqual(loss.dim(), 0)
        logger.info("Loss: %f", loss.item())

    def test_inference(self):
        logger.info("Testing the inference step.")
        x = torch.randn(4, 30, self.vae_frame_size, self.input_dim).to(self.device)
        clap = torch.randn(4, 30, self.clap_dim).to(self.device)
        prompt = torch.randn(4, 20, self.vae_frame_size, self.input_dim).to(self.device)
        lyrics = torch.randint(0, self.lyric_vocab_size, (4, 18)).to(self.device)
        with torch.no_grad():
            inp = self.music_dit_model.preprocess_input(x=x,
                                                        clap=clap,
                                                        prompt=prompt,
                                                        lyrics=lyrics)
            out = self.music_dit_model.inference(inp, cfg_scale=0.7)
        self.assertIsNotNone(out)
        self.assertEqual(out.size(), x.size())