import torch
import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.amp.rnn_compat

import torch.nn as nn

class SLSTM(nn.Module):
    def __init__(self,                 
                 dimension: int,                 
                 num_layers: int = 2,                 
                 skip: bool = True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers)
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)

    def forward(self,                
                x: torch.Tensor) -> torch.Tensor:
        x = x.permute(1, 0, 2)
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y