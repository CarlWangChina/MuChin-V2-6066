import unittest
import torch
from music_dit2.modules.diffusion import TimestepEmbedding, DiT
from music_dit2.modules.embeddings import LabelEmbedding
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger

logger = get_logger(__name__)

class TestDiT(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_timestep_embedding(self):
        logger.info("Test TimestepEmbedding")
        timestep_embedding = TimestepEmbedding(1024, device=self.device)
        self.assertIsNotNone(timestep_embedding)
        timesteps = torch.randint(0, 8000, (128,), device=self.device)
        timesteps_emb = timestep_embedding(timesteps)
        self.assertEqual(timesteps_emb.size(), (128, 1024))
        timesteps = torch.randint(0, 8000, (128, 7), device=self.device)
        timesteps_emb = timestep_embedding(timesteps)
        self.assertEqual(timesteps_emb.size(), (128, 7, 1024))

    def test_label_embedding(self):
        logger.info("Test LabelEmbedding")
        label_embedding = LabelEmbedding(1024, 128, device=self.device)
        self.assertIsNotNone(label_embedding)
        labels = torch.randint(0, 1024, (20, 6), device=self.device)
        labels_emb = label_embedding(labels)
        self.assertEqual(labels_emb.size(), (20, 6, 128))

    def test_label_embedding_with_dropout(self):
        logger.info("Test LabelEmbedding")
        label_embedding = LabelEmbedding(1024, 128, dropout=0.1, device=self.device)
        self.assertIsNotNone(label_embedding)
        labels = torch.randint(0, 1024, (20, 6), device=self.device)
        labels_emb = label_embedding(labels)
        self.assertEqual(labels_emb.size(), (20, 6, 128))
        drop_ids = torch.rand(labels.size(), dtype=torch.float, device=labels.device) < 0.1
        labels_emb = label_embedding(labels, force_drop_ids=drop_ids)
        self.assertEqual(labels_emb.size(), (20, 6, 128))

    def test_dit_with_learned_sigma(self):
        logger.info("Test DiT with leaned sigma")
        dit = DiT(input_dim=64,
                  hidden_dim=128,
                  num_layers=4,
                  num_heads=8,
                  context_dim=32,
                  use_cross_attention=True,
                  device=self.device)
        x = torch.randn(3, 128, 64).to(self.device)
        prompt = torch.randn(3, 40, 64).to(self.device)
        condition = torch.randn(3, 128, 128).to(self.device)
        context = torch.randn(3, 64, 32).to(self.device)
        timestep = torch.arange(3, device=self.device).long()
        y, variance = dit(x, timesteps=timestep, prompt=prompt, condition=condition, context=context)
        self.assertEqual(y.size(), (3, 128, 64))
        self.assertEqual(variance.size(), (3, 128, 64))
        y, variance = dit.forward_with_cfg(x, cfg_scale=0.6,
                                           timesteps=timestep, prompt=prompt, condition=condition, context=context)
        self.assertEqual(y.size(), (3, 128, 64))
        self.assertEqual(variance.size(), (3, 128, 64))

    def test_dit_without_learned_sigma(self):
        logger.info("Test DiT without leaned sigma")
        dit = DiT(input_dim=64,
                  hidden_dim=128,
                  num_layers=4,
                  num_heads=8,
                  context_dim=32,
                  use_cross_attention=True,
                  use_learned_variance=False,
                  device=self.device)
        x = torch.randn(3, 128, 64).to(self.device)
        condition = torch.randn(3, 128, 128).to(self.device)
        context = torch.randn(3, 64, 32).to(self.device)
        timestep = torch.arange(3, device=self.device).long()
        y = dit(x, timesteps=timestep, condition=condition, context=context)
        self.assertEqual(y.size(), (3, 128, 64))
        y = dit.forward_with_cfg(x, cfg_scale=0.6,
                                 timesteps=timestep, condition=condition, context=context)
        self.assertEqual(y.size(), (3, 128, 64))