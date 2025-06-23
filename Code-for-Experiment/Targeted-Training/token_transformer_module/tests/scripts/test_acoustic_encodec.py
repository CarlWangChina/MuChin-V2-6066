import unittest
from pathlib import Path
from ama_prof_divi.utils.logging import get_logger
from ama_prof_divi.configs import init_hparams, post_init_hparams
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.audioutils import get_audio_utils
from ama_prof_divi.models.acoustic.encodec import get_encodec_wrapper

logger = get_logger(__name__)
TEST_AUDIO_FILE = "10035.mp3"

class TestEncodec(unittest.TestCase):
    def setUp(self):
        init_hparams()
        post_init_hparams()
        self.encodec_wrapper = get_encodec_wrapper()
        self.au_utils = get_audio_utils()
        self.test_data_path = Path(__file__).parent.parent.joinpath('testdata')

    def test_build_encodec_wrapper(self):
        logger.info("test_build_encodec_wrapper")
        self.assertIsNotNone(self.encodec_wrapper)
        logger.info("num_channels = %d", self.encodec_wrapper.num_channels)
        logger.info("sampling_rate = %d", self.encodec_wrapper.sampling_rate)
        logger.info("codebook_size = %d", self.encodec_wrapper.codebook_size)
        logger.info("num_quantizers = %d", self.encodec_wrapper.num_quantizers)
        logger.info("segment_length = %d", self.encodec_wrapper.segment_length)
        logger.info("segment_stride = %d", self.encodec_wrapper.segment_stride)
        logger.info("bandwidth = %d", self.encodec_wrapper.bandwidth)
        logger.info("frame_rate = %d", self.encodec_wrapper.frame_rate)

    def test_encodec(self):
        logger.info("test_encodec")
        audio, sampling_rate = self.au_utils.load_audio(self.test_data_path.joinpath(TEST_AUDIO_FILE))
        self.assertIsNotNone(audio)
        audio = audio[:, :48000 * 10]

        logger.info("Loaded audio %s", TEST_AUDIO_FILE)
        audio = self.au_utils.resample(audio, sampling_rate, self.encodec_wrapper.sampling_rate, self.encodec_wrapper.num_channels)
        logger.info("Source audio sampled at %dHz: %s", self.encodec_wrapper.sampling_rate, audio.shape)
        encoded = self.encodec_wrapper.encode(audio)
        logger.info("Encoded audio: %s", encoded["audio_codes"].shape)
        logger.info("Encoded audio scales: %s", encoded["audio_scales"].shape)
        decoded = self.encodec_wrapper.decode(encoded["audio_codes"], encoded["audio_scales"])
        logger.info("Decoded audio: %s", decoded.shape)