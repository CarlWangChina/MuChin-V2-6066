import unittest
import torch
from pathlib import Path
from ama_prof_divi_common.utils import get_logger
from music_dit2.modules.transformers import Mlp, MultiHeadAttention
from Code_for_Experiment.Targeted_Training.dit_training_on_vae.music_dit2.modules.embeddings.rotary_pos_embedding import RotaryPosEmbedding

logger = get_logger(__name__)

class TestTransformers(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_mlp(self):
        logger.info("Test Mlp")
        input_dim = 32
        hidden_dim = 64
        mlp = Mlp(input_dim, hidden_dim).to(self.device)
        x = torch.randn(1, 128, input_dim).to(self.device)
        y = mlp(x)
        self.assertEqual(y.size(), (1, 128, input_dim))

    def test_mlp_with_conv(self):
        logger.info("Test Mlp with Conv")
        input_dim = 32
        hidden_dim = 64
        mlp = Mlp(input_dim, hidden_dim, use_conv=True).to(self.device)
        x = torch.randn(1, 128, input_dim).to(self.device)
        y = mlp(x)
        self.assertEqual(y.size(), (1, 128, input_dim))

    def test_self_attention(self):
        logger.info("Test Self Attention")
        input_dim = 32
        num_heads = 4
        hidden_dim = 64
        attention = MultiHeadAttention(input_dim, num_heads, hidden_dim).to(self.device)
        x = torch.randn(1, 128, input_dim).to(self.device)
        y = attention(x)
        self.assertEqual(y.size(), (1, 128, hidden_dim))

    def test_cross_attention(self):
        logger.info("Test Cross Attention")
        input_dim = 32
        num_heads = 4
        hidden_dim = 64
        context_dim = 80
        attention = MultiHeadAttention(input_dim, num_heads, hidden_dim, context_dim=80).to(self.device)
        x = torch.randn(1, 128, input_dim).to(self.device)
        c = torch.randn(1, 60, context_dim).to(self.device)
        y = attention(x, context=c)
        self.assertEqual(y.size(), (1, 128, hidden_dim))

    def test_self_attention_with_rope_and_rpr(self):
        logger.info("Test Self Attention with RoPE and RPR.")
        input_dim = 32
        num_heads = 4
        hidden_dim = 64
        rope = RotaryPosEmbedding(hidden_dim, max_position=10000).to(self.device)
        attention = MultiHeadAttention(input_dim, num_heads, hidden_dim, rope=rope, use_rpr=True).to(self.device)
        x = torch.randn(1, 128, input_dim).to(self.device)
        positions = torch.arange(128).unsqueeze(0).to(self.device)
        y = attention(x, positions=positions)
        self.assertEqual(y.size(), (1, 128, hidden_dim))