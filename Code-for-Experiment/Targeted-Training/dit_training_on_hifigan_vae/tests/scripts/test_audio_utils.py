import torch
import unittest
from pathlib import Path
from music_dit.utils import get_logger, get_audio_utils

logger = get_logger(__name__)
TEST_AUDIO_FILE = "654a3a9be052b2c2cb074230b390016697ad7f40_src.mp3"

class TestAudioUtils(unittest.TestCase):
    def setUp(self):
        self.au_utils = get_audio_utils()
        self.test_data_path = Path(__file__).parent.parent.joinpath('testdata')

    def test_audio_utils(self):
        logger.info('test_audio_utils')
        self.assertIsNotNone(self.au_utils)

    def test_audio_utils_load_audio(self):
        logger.info('test_audio_utils_load_audio')
        audio, sampling_rate = self.au_utils.load_audio(self.test_data_path.joinpath(TEST_AUDIO_FILE))
        self.assertIsNotNone(audio)
        self.assertEqual(sampling_rate, 44100)
        self.assertEqual(audio.shape, torch.Size([2, 12032409]))

    def test_audio_utils_audio_resample(self):
        logger.info('test_audio_utils_audio_resample')
        audio, sampling_rate = self.au_utils.load_audio(self.test_data_path.joinpath(TEST_AUDIO_FILE))
        audio = audio[:, :44100 * 10]
        self.assertIsNotNone(audio)
        self.assertEqual(audio.dim(), 2)
        self.assertEqual(sampling_rate, 44100)
        target_sampling_rates = [16000, 32000, 48000]
        target_num_channels_list = [1, 2]
        for target_sampling_rate in target_sampling_rates:
            for target_num_channels in target_num_channels_list:
                x = audio[0]
                self._do_test_resample(x, sampling_rate, target_sampling_rate, target_num_channels, 2)
                x = audio
                self._do_test_resample(x, sampling_rate, target_sampling_rate, target_num_channels, 2)
                x = torch.stack([audio, audio, audio], dim=0)
                self._do_test_resample(x, sampling_rate, target_sampling_rate, target_num_channels, 3)

    def _do_test_resample(self,
                          x: torch.Tensor,
                          sampling_rate: int,
                          target_sampling_rate: int,
                          target_num_channels: int,
                          output_dim):
        num_samples = x.shape[-1]
        x = self.au_utils.resample(x, sampling_rate, target_sampling_rate, target_num_channels)
        self.assertIsNotNone(x)
        self.assertEqual(x.dim(), output_dim)
        if output_dim == 3:
            self.assertEqual(x.shape[0], 3)
            self.assertEqual(x.shape[1], target_num_channels)
            self.assertEqual(x.shape[2], num_samples * target_sampling_rate // sampling_rate)
        else:
            self.assertEqual(x.shape[0], target_num_channels)
            self.assertEqual(x.shape[1], num_samples * target_sampling_rate // sampling_rate)

    def test_audio_utils_separate(self):
        logger.info("test_audio_utils_separate")
        audio, sampling_rate = self.au_utils.load_audio(self.test_data_path.joinpath(TEST_AUDIO_FILE))
        audio = audio[:, :44100 * 10]
        logger.info("demucs_sample_rate = %d", self.au_utils.demucs_sampling_rate)
        logger.info("demucs_num_channels = %d", self.au_utils.demucs_num_channels)
        self.assertEqual(audio.shape, torch.Size([2, 44100 * 10]))
        s = self.au_utils.demucs_separate(audio,
                                          sampling_rate,
                                          stems=["vocals", "other", "drums"])
        self.assertIsNotNone(s)
        self.assertEqual(s.shape, torch.Size([1, 3, 2, 44100 * 10]))
        audio = torch.stack([audio, audio, audio], dim=0)
        s = self.au_utils.demucs_separate(audio,
                                          sampling_rate,
                                          stems=["vocals", "other"])
        self.assertIsNotNone(s)
        self.assertEqual(s.shape, torch.Size([3, 2, 2, 44100 * 10]))