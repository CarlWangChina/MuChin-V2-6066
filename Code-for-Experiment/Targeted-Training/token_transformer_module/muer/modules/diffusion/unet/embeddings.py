import math
import torch
from Code_for_Experiment.Metrics.music_understanding_model.jukebox.transformer.factored_attention import repeat

def timestep_embedding(time_steps: torch.Tensor, dim: int, max_period: int = 10000, repeat_only: bool = False) -> torch.Tensor:
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=time_steps.device)
        args = time_steps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        else:
            embedding = repeat(time_steps, 'b -> b d', d=dim)
    else:
        embedding = repeat(time_steps, 'b -> b d', d=dim)
    return embedding