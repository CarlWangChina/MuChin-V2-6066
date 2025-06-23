import unittest
import torch
from ama_prof_divi_common.utils import get_logger
from Code_for_Experiment.Targeted_Training.hifigan_vocoder.ama-prof-divi_hifigan.model.discriminators import Discriminator
from ama_prof_divi_vae2.trainer.losses import *

logger = get_logger(__name__)

class TestLosses(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_kl_loss(self):
        kl_loss = get_kl_loss().to(self.device)
        mean = torch.randn(3, 512, 128).to(self.device)
        logvar = torch.randn(3, 512, 128).to(self.device)
        loss = kl_loss(mean, logvar)
        kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        logger.info("test_kl_loss: %f", loss.item())
        self.assertTrue(torch.allclose(loss, kl))

    def test_mel_loss(self):
        mel_loss = get_mel_loss().to(self.device)
        audio1 = torch.randn(3, 2, 262144).to(self.device)
        audio2 = torch.randn(3, 2, 262144).to(self.device)
        loss = mel_loss(audio1, audio2)
        logger.info("test_mel_loss: %f", loss.item())

    def test_stft_loss(self):
        stft_loss = get_stft_loss().to(self.device)
        audio1 = torch.randn(3, 2, 262144).to(self.device)
        audio2 = torch.randn(3, 2, 262144).to(self.device)
        loss = stft_loss(audio1, audio2)
        logger.info("test_stft_loss: %f", loss.item())

    def test_sisnr_loss(self):
        sisnr_loss = get_sisnr_loss().to(self.device)
        audio1 = torch.randn(3, 2, 262144).to(self.device)
        audio2 = torch.randn(3, 2, 262144).to(self.device)
        loss = sisnr_loss(audio1, audio2)
        logger.info("test_sisnr_loss: %f", loss.item())

    def test_adversarial_loss(self):
        discriminator = Discriminator().to(self.device)
        discriminator.eval()
        fake = torch.randn(3, 2, 262144).to(self.device)
        real = torch.randn(3, 2, 262144).to(self.device)
        adv_loss, gen_loss, fm_loss = discriminator(real, fake)
        logger.info("test_adversarial_loss: adv_loss=%f, gen_loss=%f, fm_loss=%f",
                    adv_loss.item(),
                    gen_loss.item(),
                    fm_loss.item())