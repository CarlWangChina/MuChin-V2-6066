import torch
import torch.nn as nn

class SLSTM(nn.Module):
    def __init__(self, 
                 dimension: int, 
                 num_layers: int = 2, 
                 skip: bool = True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers)

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y