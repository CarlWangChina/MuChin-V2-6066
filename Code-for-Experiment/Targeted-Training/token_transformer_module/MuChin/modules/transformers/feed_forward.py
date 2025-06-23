import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = -1, dropout: float = 0.1, device: str or torch.device = "cpu"):
        super(FeedForward, self).__init__()
        self.device = device
        if hidden_dim is None or hidden_dim <= 0:
            hidden_dim = dim * 4
        hidden_dim = ((hidden_dim + 255) // 256) * 256
        self.w1 = nn.Linear(dim, hidden_dim, bias=False).to(device)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False).to(device)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False).to(device)
        self.dropout_func = nn.Dropout(p=dropout).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = F.silu(self.w1(x)) + self.w3(x)
        output = self.dropout_func(output)
        output = self.w2(output)
        return output