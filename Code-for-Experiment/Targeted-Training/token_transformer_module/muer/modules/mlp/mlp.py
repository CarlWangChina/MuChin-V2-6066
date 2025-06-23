import torch
import torch.nn as nn
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.modules.mlp.mlp_args import MlpArgs

class Mlp(nn.Module):
    def __init__(self, arg: MlpArgs, device: str = "cpu"):
        super(Mlp, self).__init__()
        self.input_dim = arg.input_dim
        self.output_dim = arg.output_dim
        self.hidden_dim = self.input_dim * 2 if arg.hidden_dim is None else arg.hidden_dim
        num_layers = arg.num_layers
        dropout = arg.dropout
        activation = getattr(nn, arg.activation)
        blocks = []

        for i in range(num_layers):
            in_dim = self.input_dim if i == 0 else self.hidden_dim
            out_dim = self.output_dim if i == num_layers - 1 else self.hidden_dim
            blocks.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                blocks.append(activation())
                if dropout is not None:
                    blocks.append(nn.Dropout(p=dropout))
        self.mlp = nn.Sequential(*blocks).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)