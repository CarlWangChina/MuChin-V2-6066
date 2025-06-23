import unittest
from pathlib import Path
import math
import torch
from music_dit.utils import get_logger, get_audio_utils, get_hparams
from music_dit.modules.vae import VAEPreprocessor, VAEModel
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.modules.vae.training.model import VAEModelForTraining
from music_dit.modules.vae.training.losses import (
    MultiScaleMelSpectrogramLoss,
    MRSTFTLoss,
    SISNRLoss,
    get_adversarial_loss
)

logger = get_logger(__name__)
TEST_AUDIO_FILE = "654a3a9be052b2c2cb074230b390016697ad7f40_src.mp3"

class TestVAE(unittest.TestCase):
    def setUp(self):
        self.au_utils = get_audio_utils()
        self.vae_preprocessor = VAEPreprocessor(self.au_utils)
        self.test_data_path = Path(__file__).parent.parent.joinpath('testdata')
        torch.backends.cudnn.enabled = False

    def test_preprocess(self):
        logger.info("test_preprocess")
        audio, sr = self.au_utils.load_audio(self.test_data_path.joinpath(TEST_AUDIO_FILE))
        inp = self.vae_preprocessor(audio, sr)
        self.assertIsNotNone(inp["input_values"])
        self.assertIsNotNone(inp["padding_mask"])
        logger.info("input_values.shape = %s", inp["input_values"].shape)
        logger.info("padding_mask.shape = %s", inp["padding_mask"].shape)
        assert inp["input_values"].size() == (1, audio.size(0), inp["input_values"].shape[2])
        assert inp["padding_mask"].size() == (1, inp["input_values"].shape[2])

    def test_vae_model(self):
        logger.info("test_vae_model")
        vae_model = VAEModel()
        self.assertIsNotNone(vae_model)
        logger.info("vae_model: frame_size=%d, embedding_dim=%d, time_unit=%f sec, frame_rate=%f fps",
                    vae_model.frame_size, vae_model.embedding_dim,
                    vae_model.time_unit, vae_model.frame_rate)
        logger.info("Testing encoder and decoder")
        audio, sr = self.au_utils.load_audio(self.test_data_path.joinpath(TEST_AUDIO_FILE))
        inp = self.vae_preprocessor(audio, sr)
        logger.info("Input audio.shape = %s", inp["input_values"].shape)
        with torch.no_grad():
            encoded = vae_model.encode(inp["input_values"], inp["padding_mask"])
            self.assertIsNotNone(encoded)
            logger.info("Encoded shape = %s", encoded.shape)
            audio_output = vae_model.decode(encoded)
            self.assertIsNotNone(audio_output)
            logger.info("Decoded audio_output shape = %s", audio_output.shape)
            self.assertEqual(audio_output.size(0), inp["input_values"].size(0))
            self.assertEqual(audio_output.size(1), inp["input_values"].size(1))
            self.assertEqual(math.floor(audio_output.size(2) / vae_model.chunk_stride),
                             math.ceil(inp["input_values"].size(2) / vae_model.chunk_stride))

    def test_vae_training_model(self):
        logger.info("test_vae_training_model")
        model = VAEModelForTraining()
        x = torch.randn(10, model.num_channels, model.chunk_length).to(model.device)
        with torch.no_grad():
            y = model.encode_decode(x)
            self.assertIsNotNone(y)
            self.assertEqual(y.size(), x.size())
            logits, fmaps = model.discriminate(x)
            self.assertIsNotNone(logits)
            self.assertIsNotNone(fmaps)
            self.assertEqual(len(logits), len(fmaps))

    def test_losses(self):
        logger.info("test_losses")
        device = get_hparams().device
        x = torch.randn(10, 2, 46080).to(device)
        y = torch.randn(10, 2, 46080).to(device)
        msspec_loss = MultiScaleMelSpectrogramLoss(sampling_rate=44100,
                                                   range_start=9,
                                                   range_end=14).to(device)
        loss = msspec_loss(x, y)
        self.assertEqual(loss.dim(), 0)
        logger.info("MultiScaleMelSpectrogramLoss: loss=%f", loss)
        mrstft_loss = MRSTFTLoss().to(device)
        loss = mrstft_loss(x, y)
        self.assertEqual(loss.dim(), 0)
        logger.info("MRSTFTLoss: loss=%f", loss)
        sisnr_loss = SISNRLoss(sample_rate=44100).to(device)
        loss = sisnr_loss(x, y)
        self.assertEqual(loss.dim(), 0)
        logger.info("MRSTFTLoss: loss=%f", loss)
        model = VAEModelForTraining()
        adv_loss = get_adversarial_loss(model, device)
        d_loss, feat_loss, fake_loss, real_loss = adv_loss(x, y)
        self.assertEqual(d_loss.dim(), 0)
        self.assertEqual(feat_loss.dim(), 0)
        self.assertEqual(fake_loss.dim(), 0)
        self.assertEqual(real_loss.dim(), 0)
        logger.info("AdversarialLoss: d_loss=%f, g_loss=%f, fake_loss=%f, real_loss=%f",
                    d_loss, feat_loss, fake_loss, real_loss)