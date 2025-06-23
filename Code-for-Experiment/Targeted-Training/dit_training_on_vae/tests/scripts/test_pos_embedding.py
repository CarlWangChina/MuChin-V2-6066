import torch
import unittest
from music_dit2.modules.embeddings import SinusoidalPosEmbedding, RotaryPosEmbedding
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger
logger = get_logger(__name__)

class TestPositionalEmbedding(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sinusoidal_embedding = SinusoidalPosEmbedding(dim=32, max_position=512, device=self.device)
        self.rotary_embedding = RotaryPosEmbedding(dim=32, max_position=512, device=self.device)
        self.assertEqual(self.sinusoidal_embedding.weight.size(), torch.Size([1, 512, 32]))
        self.assertEqual(self.rotary_embedding.freqs_cis.size(), torch.Size([512, 16, 2]))

    def test_sinusoidal_embedding(self):
        logger.info("Testing sinusoidal positional embedding ...")
        self.assertIsNotNone(self.sinusoidal_embedding)
        pos = torch.tensor([1, 2, 3, 4, 5], device=self.device)
        pos_embedding = self.sinusoidal_embedding(pos)
        self.assertEqual(pos_embedding.size(), torch.Size([1, 5, 32]))
        pos = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], device=self.device)
        pos_embedding = self.sinusoidal_embedding(pos)
        self.assertEqual(pos_embedding.size(), torch.Size([2, 5, 32]))
        pos = torch.tensor([0])
        pos_embedding = self.sinusoidal_embedding(pos)
        self.assertEqual(pos_embedding.size(), torch.Size([1, 1, 32]))
        for i in range(32):
            if i % 2 == 0:
                self.assertEqual(pos_embedding[0, 0, i], torch.Tensor([0.0]).to(self.device))
            else:
                self.assertEqual(pos_embedding[0, 0, i], torch.Tensor([1.0]).to(self.device))

    def test_rotary_embedding(self):
        logger.info("Test rotary positional embedding ...")
        self.assertIsNotNone(self.rotary_embedding)
        pos = torch.tensor([1, 2, 3, 4, 5], device=self.device)
        x = torch.randn(1, 5, 32, device=self.device)
        pos_embedding = self.rotary_embedding(x, pos)
        self.assertEqual(pos_embedding.size(), torch.Size([1, 5, 32]))
        pos = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], device=self.device)
        x = torch.randn(2, 5, 32, device=self.device)
        pos_embedding = self.rotary_embedding(x, pos)
        self.assertEqual(pos_embedding.size(), torch.Size([2, 5, 32]))