import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.amp.rnn_compat
import torch.nn as nn

class ApproxGELU(nn.Module):
    def __init__(self, approximate: str = "tanh"):
        super().__init__()
        self.gelu = nn.GELU(approximate=approximate)

    def forward(self, x):
        return self.gelu(x)