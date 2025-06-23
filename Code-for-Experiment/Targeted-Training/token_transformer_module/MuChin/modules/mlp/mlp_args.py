from dataclasses import dataclass
from typing import Optional

@dataclass
class MlpArgs:
    input_dim: int
    output_dim: int
    num_layers: int = 2
    hidden_dim: Optional[int] = None
    dropout: Optional[float] = None
    activation: str = 'ReLU'