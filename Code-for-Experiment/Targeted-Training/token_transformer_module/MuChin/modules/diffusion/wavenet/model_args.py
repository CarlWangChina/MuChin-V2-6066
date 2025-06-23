from dataclasses import dataclass

@dataclass
class WaveNetModelArgs:
    in_channels: int
    out_channels: int
    model_channels: int
    context_channels: int
    num_layers: int
    dilation_cycle: int = 4
    dims: int = 1