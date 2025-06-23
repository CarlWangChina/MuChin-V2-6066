import torch
import torch.nn as nn
import torch.nn.functional as F
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger
logger = get_logger(__name__)

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size: int, dim: int, local_rank: int = 0):
        super(EmbeddingModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, dim)
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, vocab_size)
        self.local_rank = local_rank
        if self.local_rank == 0:
            logger.info(f"EmbeddingModel created: vocab_size={vocab_size}, dim={dim}")

    def forward(self, x):
        assert 0 <= x.min() <= x.max() <= self.vocab_size
        x = self.embedding(x)
        x = F.relu(x)
        x = self.linear(x)
        x = F.relu(x)
        logits = self.out_proj(self.norm(x))
        return logits