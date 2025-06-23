from pathlib import Path
import unittest
import torch
from tqdm import tqdm
from ama_prof_divi.utils.logging import get_logger
from ama_prof_divi.configs import init_hparams, post_init_hparams, get_hparams
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.audioutils import get_audio_utils
import get_audio_utils
import get_svs_model
logger = get_logger(__name__)
TEST_AUDIO_FILE = "10035.mp3"

class TestSVS(unittest.TestCase):
    def setUp(self):
        self.hparams = init_hparams()
        post_init_hparams()
        self.svs = get_svs_model()
        self.au_utils = get_audio_utils()
        self.phoneme_tokenizer = self.svs.phoneme_tokenizer
        self.device = get_hparams()["ama-prof-divi"]["device"]
        self.test_data_path = Path(__file__).parent.parent.joinpath('testdata')

    def test_svs_model(self):
        logger.info("test_svs_model")
        self.assertIsNotNone(self.svs)

    @torch.inference_mode()
    def _test_svs_vocoder(self):
        logger.info("test_svs_vocoder")
        self.assertIsNotNone(self.svs.vocoder)
        audio, sampling_rate = self.au_utils.load_audio(self.test_data_path.joinpath(TEST_AUDIO_FILE))
        logger.info("audio.shape: {}".format(audio.shape))
        audio = self.au_utils.demucs_separate(audio, sampling_rate, ["vocals"]).squeeze(0).squeeze(0)
        audio = self.au_utils.resample(audio, self.au_utils.demucs_sampling_rate, self.svs.vocoder.sampling_rate, 2)
        hop_size = self.svs.vocoder.hop_size
        window_size = ((sampling_rate * 2 + hop_size - 1) // hop_size) * hop_size
        num_windows = (audio.shape[1] + window_size - 1) // window_size
        output = torch.zeros((audio.shape[0], 0), dtype=audio.dtype, device="cpu")
        for i in tqdm(range(num_windows)):
            audio_win = audio[:, window_size * i:window_size * (i + 1)]
            spec = self.svs.vocoder.mel_encode(audio_win.to(self.device))
            y = self.svs.vocoder(spec).to("cpu")
            if i != num_windows - 1:
                self.assertEqual(audio_win.shape, y.shape)
            output = torch.cat((output, y), dim=1)
        logger.info("output.shape: {}".format(output.shape))

    @torch.inference_mode()
    def test_svs_generator(self):
        logger.info("test_svs_generator")
        phoneme_tokens = torch.randint(0, self.phoneme_tokenizer.vocab_size, (2, 100)).to(self.device)
        pitch_tokens = torch.randint(0, 127, (2, 800)).to(self.device)
        mel = self.svs.generate(phoneme_tokens, pitch_tokens)
        logger.info("mel.shape: {}".format(mel.shape))