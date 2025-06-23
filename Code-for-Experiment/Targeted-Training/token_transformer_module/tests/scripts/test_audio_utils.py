import torch
import unittest
from pathlib import Path
from ama_prof_divi.utils import logging
from ama_prof_divi.configs import init_hparams, post_init_hparams
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.audioutils import get_audio_utils
_utilslogger = logging.getLogger(__name__)
TEST_AUDIO_FILE = "10035.mp3"

class TestAudioUtils(unittest.TestCase):
    def setUp(self):
        init_hparams()
        post_init_hparams()
        self.au_utils = get_audio_utils()
        self.test_data_path = Path(__file__).parent.parent.joinpath('testdata')

    def test_audio_utils(self):
        _utilslogger.info('test_audio_utils')
        self.assertIsNotNone(self.au_utils)
        _utilslogger.info("Audio utils: %s", self.au_utils)

    def test_audio_utils_load_audio(self):
        _utilslogger.info('test_audio_utils_load_audio')
        audio, sampling_rate = self.au_utils.load_audio(self.test_data_path.joinpath(TEST_AUDIO_FILE))
        self.assertIsNotNone(audio)
        self.assertEqual(sampling_rate, 44100)
        self.assertEqual(audio.shape, torch.Size([2, 11239149]))

    def test_audio_utils_audio_resample(self):
        _utilslogger.info('test_audio_utils_audio_resample')
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

    def _do_test_resample(self, x: torch.Tensor, sampling_rate: int, target_sampling_rate: int, target_num_channels: int, output_dim):
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
        _utilslogger.info("test_audio_utils_separate")
        audio, sampling_rate = self.au_utils.load_audio(self.test_data_path.joinpath(TEST_AUDIO_FILE))
        audio = audio[:, :44100 * 10]

        self.assertEqual(audio.shape, torch.Size([2, 44100 * 10]))
        s = self.au_utils.demucs_separate(audio, sampling_rate, stems=["vocals", "other", "drums"])
        self.assertIsNotNone(s)
        self.assertEqual(s.shape, torch.Size([1, 3, 2, 44100 * 10]))
        audio = torch.stack([audio, audio, audio], dim=0)
        s = self.au_utils.demucs_separate(audio, sampling_rate, stems=["vocals", "other"])
        self.assertIsNotNone(s)
        self.assertEqual(s.shape, torch.Size([3, 2, 2, 44100 * 10]))