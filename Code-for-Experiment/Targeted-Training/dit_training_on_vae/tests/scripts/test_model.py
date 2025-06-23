import unittest
import torch
from ama_prof_divi_common.utils import get_logger
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.models.music_dit import MusicDiTModel

logger = get_logger(__name__)

class TestMusicDiTModel(unittest.TestCase):
    def setUp(self):
        self.model = MusicDiTModel()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def test_training_step(self):
        logger.info("Testing training step...")
        logger.info("Lyrics vocab size: %d" % self.model.lyrics_vocab_size)
        vae = torch.randn((3, 100, 512)).to(self.device)
        vae_mask = torch.ones((3, 100)).to(self.device)
        clap = torch.randn((3, 12, 512)).to(self.device)
        clap_mask = torch.ones((3, 12)).to(self.device)
        lyrics = torch.randint(0, self.model.lyrics_vocab_size, (3, 90)).to(self.device)
        lyrics_mask = torch.ones((3, 90)).to(self.device)
        with torch.no_grad():
            loss = self.model(
                vae=vae,
                vae_mask=vae_mask,
                clap=clap,
                clap_mask=clap_mask,
                lyrics=lyrics,
                lyrics_mask=lyrics_mask
            )
            logger.info("Loss=%s" % loss)

    def test_inference(self):
        logger.info("Test inference...")
        x = torch.randn((3, 100, 512)).to(self.device)
        mask = torch.ones((3, 100)).to(self.device)
        clap = torch.randn((3, 12, 512)).to(self.device)
        clap_mask = torch.ones((3, 12)).to(self.device)
        lyrics = torch.randint(0, self.model.lyrics_vocab_size, (3, 90)).to(self.device)
        lyrics_mask = torch.ones((3, 90)).to(self.device)
        with torch.no_grad():
            generated = self.model.inference(
                x=x,
                mask=mask,
                clap=clap,
                clap_mask=clap_mask,
                lyrics=lyrics,
                lyrics_mask=lyrics_mask
            )
            self.assertEqual(generated.size(), x.size())

    def test_model_size(self):
        logger.info("Test model size...")
        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info("Number of parameters of the DI model: %d" % num_params)