import torch
import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.amp.rnn_compat
from typing import Optional, Union, List, Dict
from music_dit.utils import get_logger, get_hparams, get_audio_utils, AudioUtils

logger = get_logger(__name__)

class VAEPreprocessor(torch.nn.Module):
    def __init__(self, 
                 au_utils: Optional[AudioUtils] = None, 
                 normalize_loudness: Optional[bool] = None, 
                 input_loudness: Optional[float] = None):
        super(VAEPreprocessor, self).__init__()
        self.au_utils = au_utils if au_utils is not None else get_audio_utils()
        hparams = get_hparams()
        self.sampling_rate = hparams.vae.sampling_rate
        self.num_channels = hparams.vae.num_channels
        self.default_device = hparams.device
        self.normalize_loudness = normalize_loudness if normalize_loudness is not None else hparams.vae.normalize_loudness
        self.input_loudness = input_loudness if input_loudness is not None else hparams.vae.input_loudness

    @torch.no_grad()
    def preprocess(self, 
                   audio: Union[torch.Tensor, List[torch.Tensor]], 
                   sampling_rate: int, 
                   device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        audio_list = audio if isinstance(audio, list) else [audio]
        if device is None:
            device = self.default_device
        batch_size = len(audio_list)
        output_list = []
        max_num_samples = 0
        for batch, x in enumerate(audio_list):
            x = x.cpu()
            assert x.dim() == 2, ("Only 2-dim audio samples are supported. Shape should be (num_channels, num_samples).")
            if sampling_rate != self.sampling_rate or x.size(0) != self.num_channels:
                x = self.au_utils.resample(x, sampling_rate, self.sampling_rate, self.num_channels)
            if self.normalize_loudness:
                x = self.au_utils.normalize_loudness(x, self.sampling_rate, self.input_loudness)
            output_list.append(x)
            max_num_samples = max(max_num_samples, x.size(1))
        input_values = torch.zeros(batch_size, self.num_channels, max_num_samples, device=device)
        padding_mask = torch.zeros(batch_size, max_num_samples, device=device).long()
        for batch, x in enumerate(output_list):
            input_values[batch, :, :x.size(1)] = x.to(device)
            padding_mask[batch, :x.size(1)] = 1
        return {"input_values": input_values,
                "padding_mask": padding_mask,
                "sampling_rate": self.sampling_rate}

    def forward(self, 
                audio: Union[torch.Tensor, List[torch.Tensor]], 
                sampling_rate: int, 
                device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        return self.preprocess(audio, 
                               sampling_rate=sampling_rate, 
                               device=device)