import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.amp.rnn_compat
from typing import Optional, Type
import torch.nn as nn
from functools import partial

class Mlp(nn.Module):
    def __init__(self,                 input_dim: int,                 hidden_dim: int,                 output_dim: Optional[int] = None,                 act_layer: Type[nn.Module] = nn.GELU,                 use_bias: bool = True,                 use_conv: bool = False,                 dropout: float = 0.0):
        super(Mlp, self).__init__()
        self.use_conv = use_conv
        self.dropout = dropout
        output_dim = output_dim or input_dim
        linear_layer = partial(nn.Conv1d, kernel_size=1) if use_conv else nn.Linear
        self.fc1 = linear_layer(input_dim, hidden_dim, bias=use_bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = linear_layer(hidden_dim, output_dim, bias=use_bias)
        self.drop2 = nn.Dropout(dropout)

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        if self.use_conv:
            x = x.transpose(-2, -1)
        x = self.fc1(x)
        x = self.act(x)
        if self.training and self.dropout > 0.0:
            x = self.drop1(x)
        x = self.fc2(x)
        if self.training and self.dropout > 0.0:
            x = self.drop2(x)
        if self.use_conv:
            x = x.transpose(-2, -1)
        return x