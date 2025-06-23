import torch
import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.amp.rnn
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, output_dim: int, dim: int = -1, device: str or torch.device = "cpu"):
        super(LayerNorm, self).__init__()
        self.dim = dim
        self.layer_norm = nn.LayerNorm(output_dim, device=device)

    def forward(self, x):
        if self.dim == -1:
            return self.layer_norm(x)
        else:
            x = x.transpose(self.dim, -1)
            x = self.layer_norm(x)
            x = x.transpose(self.dim, -1)
            return x

class PredictorBase(nn.Module):
    def __init__(self, dim: int, num_layers: int = 2, num_channels: int = -1, kernel_size: int = 3, dropout: float = 0.1, padding_type: str = "same", device: str = "cpu"):
        super(PredictorBase, self).__init__()
        self.device = device
        self.dim = dim
        if num_channels == -1:
            num_channels = dim
        self.num_channels = num_channels
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = dim if i == 0 else num_channels
            self.conv_layers.append(
                nn.Sequential(
                    nn.ConstantPad1d(padding=((kernel_size - 1) // 2, (kernel_size - 1) // 2) if padding_type == 'same' else (kernel_size - 1, 0), value=0),
                    nn.Conv1d(in_channels=in_channels, out_channels=num_channels, kernel_size=kernel_size, stride=1, padding=0, device=device),
                    nn.ReLU(),
                    LayerNorm(num_channels, dim=1, device=device),
                    nn.Dropout(dropout)
                )
            )