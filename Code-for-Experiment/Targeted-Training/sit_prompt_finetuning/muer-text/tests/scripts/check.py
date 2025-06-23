import torch
import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.amp.rnn_compat
from typing import Optional
import torch.nn as nn

class LabelEmbedding(nn.Module):
    def __init__(self,
                 num_classes: int,
                 hidden_dim: int,
                 *,
                 dropout: float = 0.0,
                 device: Optional[torch.device] = None):
        super(LabelEmbedding, self).__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.embed = nn.Embedding(num_classes + 1 if dropout > 0 else num_classes,
                                  hidden_dim)
        if device is not None:
            self.to(device)

    def token_drop(self,
                   labels: torch.Tensor,
                   *,
                   force_drop_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape, device=labels.device) < self.dropout
        else:
            drop_ids = (force_drop_ids != 0)
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self,
                labels: torch.Tensor,
                *,
                force_drop_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        labels = labels.long()
        assert 0 <= labels.min() and labels.max() < self.num_classes, \
            f"Labels should be in the range [0, {self.num_classes - 1}]"
        if self.training and self.dropout > 0:
            labels = self.token_drop(labels,
                                     force_drop_ids=force_drop_ids)
        return self.embed(labels)

if __name__ == '__main__':
    with open('phoneme_tokens.txt', 'r') as f:
        phoneme_vocab = f.read().splitlines()
    vocab = {note: i for i, note in enumerate(phoneme_vocab)}
    print(vocab)
    vocab_size = len(vocab)
    hidden_dim = 256
    lyrics_embedding = LabelEmbedding(num_classes=vocab_size,
                                      hidden_dim=hidden_dim)
    lyrics = 'C.Q C.ING C.ZH C.U C.SH C.ENG C.L C.I C.G C.UO C.X C.IN C.CH C.UN'
    lyrics_labels = torch.tensor([vocab[note] for note in lyrics.split()], dtype=torch.long)
    embeddings = lyrics_embedding(lyrics_labels)
    print(lyrics_labels)
    print(embeddings.shape)