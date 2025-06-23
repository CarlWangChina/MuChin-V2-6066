from dataclasses import dataclass
from typing import Optional

@dataclass
class TransformerModelArgs:
    dim: int = 768
    num_layers: int = 8
    num_heads: int = 8
    num_quantization_groups: int = 1
    dropout: float = 0.1
    max_seq_len: int = 2048
    max_position_embeddings: Optional[int] = None
    hidden_dim: Optional[int] = None
    vocab_size: int = -1
    kv_dim: Optional[int] = None
    pos_embedding: str = "rotary"