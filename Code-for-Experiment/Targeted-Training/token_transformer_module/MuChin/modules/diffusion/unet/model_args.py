from dataclasses import dataclass

@dataclass
class UnetModelArgs:
    in_channels: int = 320
    out_channels: int = 320
    model_channels: int = 320
    context_dim: int = -1
    num_res_blocks: int = 2
    attention_resolutions: list[int] = [4, 2, 1]
    dropout: float = 0.0
    channel_mult: list[int] = [1, 2, 4, 4]
    conv_resample: bool = True
    dims: int = 1
    num_heads: int = 1
    use_transformer: bool = False
    transformer_depth: int = 1
    use_scale_shift_norm: bool = False
    res_block_updown: bool = False
    use_time_embedding: bool = True
    use_controlnet: bool = False