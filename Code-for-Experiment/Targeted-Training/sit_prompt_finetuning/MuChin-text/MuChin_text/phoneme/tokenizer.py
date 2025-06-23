import torch
import torch.nn as nn
from pathlib import Path
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger

logger = get_logger(__name__)

class PhonemeTokenizer(nn.Module):
    def __init__(self):
        super(PhonemeTokenizer, self).__init__()
        self.phoneme2idx = {}
        token_file = Path(__file__).parent.parent.parent / "configs" / "phoneme_tokens.txt"
        with open(token_file, "r", encoding="utf-8") as f:
            idx = 0
            for line in f:
                line = line.strip()
                if line != "" and not line.startswith("#"):
                    self.phoneme2idx[line] = idx
                    idx += 1
        self.unknown_token_id = idx
        self.idx2phoneme = {v: k for k, v in self.phoneme2idx.items()}
        self.vocab_size = idx + 1

    def tokenize(self, phoneme: str) -> torch.Tensor:
        tokens = phoneme.split()
        token_ids = []
        for t in tokens:
            if t in self.phoneme2idx:
                token_ids.append(self.phoneme2idx[t])
            else:
                logger.warning("Unknown phoneme: %s", t)
                token_ids.append(self.unknown_token_id)
        return torch.tensor(token_ids, dtype=torch.long)

    def forward(self, phoneme: str) -> torch.Tensor:
        return self.tokenize(phoneme)