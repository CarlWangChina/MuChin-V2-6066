import torch
import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.amp.rnn as rnn
import torch.nn.functional as F
from typing import Optional

class LengthRegulator(torch.nn.Module):
    """Length regulator module.
    Args:
        padding_value (float): Value used for padding.
    """

    def __init__(self, padding_value: int = 0):
        super(LengthRegulator, self).__init__()
        self.padding_value = padding_value

    def forward(self, duration: torch.Tensor, *, padding_mask: Optional[torch.Tensor] = None, rate_ratio: float = 1.0) -> (torch.Tensor, torch.Tensor):
        """Length regulate.
        Example (no batch dim version):
            1. dur = [2,2,3]
            2. token_idx = [[1],[2],[3]], dur_cum_sum = [2,4,7], dur_cum_sum_prev = [0,2,4]
            3. token_mask = [[1,1,0,0,0,0,0],[0,0,1,1,0,0,0],[0,0,0,0,1,1,1]]
            4. token_idx * token_mask = [[1,1,0,0,0,0,0],[0,0,2,2,0,0,0],[0,0,0,0,3,3,3]]
            5. (token_idx * token_mask).sum(0) = [1,1,2,2,3,3,3]

        Args:
            duration (torch.Tensor): Durations of each frame in phonemes: (num_batches, seq_len).
            padding_mask (torch.Tensor): Padding mask tensor: (num_batches, seq_len).
            rate_ratio (float): The ratio of output (MEL) feature rate to the melody feature rate.
        Returns:
            mel_duration (torch.Tensor): Length regulated durations in MELs: (num_batches, seq_len).
            dur_cum_sum (torch.Tensor): Cumulative sum of durations in phonemes: (num_batches, seq_len).
        """

        assert rate_ratio > 0.0
        duration = torch.round(duration.float().cpu() * rate_ratio).long()
        if padding_mask is not None:
            duration *= padding_mask.long().cpu()
        token_idx = torch.arange(1, duration.shape[1] + 1)[None, :, None]
        dur_cum_sum = torch.cumsum(duration, dim=1)
        dur_cum_sum_prev = F.pad(dur_cum_sum, [1, -1], mode='constant', value=0)
        pos_idx = torch.arange(duration.sum(-1).max())[None, None]
        token_mask = (pos_idx >= dur_cum_sum_prev[:, :, None]) & (pos_idx < dur_cum_sum[:, :, None])
        mel_duration = (token_idx * token_mask.long()).sum(1)
        if self.padding_value != 0:
            mel_duration[mel_duration == 0] = self.padding_value
        return mel_duration, dur_cum_sum