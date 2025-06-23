import unittest
import torch
import torch.nn.functional as F
from ama_prof_divi_common.utils import get_logger
from ama_prof_divi_common.utils.dist_wrapper import get_device
from ama_prof_divi_hifigan.model import build_generator, build_discriminator
from Code_for_Experiment.Targeted_Training.hifigan_vocoder.ama-prof-divi_hifigan.mel import get_mel_generator

logger = get_logger(__name__)

class TestHifiGAN(unittest.TestCase):
    def setUp(self):
        self.device = get_device()

    def test_generator(self):
        logger.info("Test Generator")
        for version in ["v1", "v2", "v3", "v1_large"]:
            generator = build_generator(version)
            self.assertIsNotNone(generator)
            self.assertEqual(generator.version, version)

    def test_inference(self):
        logger.info("Test inference")
        generator = build_generator("v1_large").to(self.device)
        generator.eval()
        generator.remove_weight_norm_()
        hop_length = generator.hparams.mel_default.hop_length
        with torch.no_grad():
            mel = torch.randn(1, 160, 100).to(self.device)
            audio = generator(mel).squeeze(1)
            self.assertEqual(audio.size(0), 1)
            self.assertEqual(audio.size(1), mel.size(-1) * hop_length)

    def test_training(self):
        logger.info("Test training")
        generator = build_generator("v1_large").to(self.device)
        discriminator = build_discriminator().to(self.device)
        mel_generator = get_mel_generator(generator).to(self.device)
        generator.train()
        discriminator.train()
        y = torch.randn(2, 1, 32768).to(self.device)
        x = mel_generator(y).squeeze(1)
        self.assertEqual(x.size(), (2, 160, x.size(-1)))
        y_g_hat = generator(x)
        y_g_hat_mel = mel_generator(y_g_hat).squeeze(1)
        self.assertEqual(y_g_hat_mel.size(), x.size())
        loss_mel = F.l1_loss(x, y_g_hat_mel) * 45
        loss_d, _, _ = discriminator(y, y_g_hat.detach())
        loss_d.backward()
        _, loss_g, loss_fm = discriminator(y, y_g_hat)
        loss_g = loss_g + loss_fm + loss_mel
        loss_g.backward()