import unittest
import torch
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger
from ama_prof_divi_vae2.model import VAE2Model

logger = get_logger(__name__)

class TestVAEInference(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VAE2Model().to(self.device)
        self.model.remove_weight_norm_()
        self.model.eval()

    def test_model_size(self):
        encoder_size = sum(p.numel() for p in self.model.encoder.parameters())
        decoder_size = sum(p.numel() for p in self.model.decoder.parameters())
        logger.info("test_model_size: encoder_size: %s, decoder_size: %s", encoder_size, decoder_size)

    def test_encode_frame(self):
        audio_frame = torch.randn(3, 2, self.model.chunk_length).to(self.device)
        with torch.no_grad():
            mean, log_var = self.model.encode_frame(audio_frame)
        logger.info("test_encode_frame: audio_frame: %s --> mean: %s, logvar: %s", audio_frame.size(), mean.size(), log_var.size())
        self.assertEqual(mean.size(), log_var.size())
        self.assertEqual(mean.size(), (audio_frame.size(0), self.model.dim, audio_frame.size(-1) // self.model.hop_length))

    def test_decode_frame(self):
        latent_frame = torch.randn(3, self.model.dim, self.model.frame_size).to(self.device)
        with torch.no_grad():
            audio_frame = self.model.decode_frame(latent_frame)
        self.assertEqual(audio_frame.size(), (3, 2, self.model.chunk_length))

    def test_encode_decode(self):
        audio_stream = torch.randn(3, 2, self.model.chunk_length * 3).to(self.device)
        with torch.no_grad():
            mean, logvar = self.model.encode(audio_stream)
            decoded_stream = self.model.decode(mean)
        logger.info("test_encode_decode: audio_stream: %s --> mean: %s --> decoded_stream: %s", audio_stream.size(), logvar.size(), decoded_stream.size())

    def test_forward(self):
        audio_frame = torch.randn(3, 2, self.model.chunk_length).to(self.device)
        with torch.no_grad():
            recon, mean, log_var = self.model(audio_frame)
        self.assertEqual(recon.size(), audio_frame.size())
        self.assertEqual(mean.size(), log_var.size())
        logger.info("test_forward: done.")