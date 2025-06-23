import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from pathlib import Path
from ama_prof_divi_common.utils import get_logger, get_hparams
from ..modules.seanet import SEANetDecoder, SEANetEncoder
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.misc import linear_overlap_add

logger = get_logger(__name__)

class VAE2Model(nn.Module):
    def __init__(self, training: bool = False):
        super(VAE2Model, self).__init__()
        root_path = Path(__file__).parent.parent.parent
        hparams = get_hparams(root_path).vae
        hparams_training = get_hparams().training
        if training:
            if "dropout_rate" in hparams_training.trainer:
                hparams.dropout_rate = hparams_training.trainer.dropout_rate
        if "dropout_rate" not in hparams:
            hparams.dropout_rate = 0.0
        self.encoder = SEANetEncoder(num_channels=hparams.num_channels,
                                     latent_dim=hparams.dim,
                                     num_initial_filters=hparams.num_filters,
                                     upsampling_ratios=hparams.upsampling_ratios,
                                     upsampling_kernel_sizes=hparams.upsampling_kernel_sizes,
                                     resblock_kernel_sizes=hparams.resblock_kernel_sizes,
                                     resblock_dilation_sizes=hparams.resblock_dilation_sizes,
                                     initial_kernel_size=hparams.initial_kernel_size,
                                     last_kernel_size=hparams.last_kernel_size,
                                     num_lstm_layers=hparams.num_lstm_layers,
                                     activation=hparams.activation,
                                     activation_params=hparams.activation_params,
                                     final_activation=hparams.final_activation,
                                     norm=hparams.norm_type,
                                     norm_params=hparams.norm_params,
                                     dropout=hparams.dropout_rate)
        self.decoder = SEANetDecoder(num_channels=hparams.num_channels,
                                     latent_dim=hparams.dim,
                                     num_initial_filters=hparams.num_filters,
                                     upsampling_ratios=hparams.upsampling_ratios,
                                     upsampling_kernel_sizes=hparams.upsampling_kernel_sizes,
                                     resblock_kernel_sizes=hparams.resblock_kernel_sizes,
                                     resblock_dilation_sizes=hparams.resblock_dilation_sizes,
                                     initial_kernel_size=hparams.initial_kernel_size,
                                     last_kernel_size=hparams.last_kernel_size,
                                     num_lstm_layers=hparams.num_lstm_layers,
                                     activation=hparams.activation,
                                     activation_params=hparams.activation_params,
                                     final_activation=hparams.final_activation,
                                     norm=hparams.norm_type,
                                     norm_params=hparams.norm_params,
                                     dropout=hparams.dropout_rate)
        self.sampling_rate = hparams.sampling_rate
        self.num_channels = hparams.num_channels
        self.dim = hparams.dim
        self.chunk_length = hparams.chunk_length
        self.chunk_stride = hparams.chunk_stride
        self.hop_length = int(np.prod(hparams.upsampling_ratios))
        assert self.chunk_length % self.hop_length == 0, \
            f"chunk_length ({self.chunk_length}) must be divisible by the hop length ({self.hop_length})"
        self.frame_size = self.chunk_length // self.hop_length
        self.time_unit = self.chunk_stride / self.sampling_rate

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self,
               audio: torch.Tensor,
               padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        assert audio.dim() == 3, ("Only 3-dim audio samples are supported. Shape should be "
                                  "(batch_size, num_channels, num_samples).")
        assert audio.size(1) == self.num_channels, \
            (f"num_channels ({self.num_channels}) does not match the second dimension of audio ({audio.size(1)})."
             f"Please use the preprocess method to resample the audio samples to {self.num_channels} channels.")
        if padding_mask is not None:
            assert padding_mask.dim() == 2, ("Only 2-dim padding mask is supported. Shape should be "
                                             "(batch_size, num_samples).")
            assert padding_mask.size() == (audio.size(0), audio.size(2)), \
                (f"padding_mask.shape ({padding_mask.size()}) does not match audio.shape ({audio.size()}). "
                 f"Should be (batch_size, num_samples).")
            audio = audio * padding_mask.unsqueeze(1)
        seq_length = audio.size(2)
        encoded_mean_frames = []
        encoded_logvar_frames = []
        for offset in range(0, seq_length, self.chunk_stride):
            audio_chunk = audio[:, :, offset:offset + self.chunk_length]
            mean, logvar = self.encode_frame(audio_chunk)
            encoded_mean_frames.append(mean)
            encoded_logvar_frames.append(logvar)
        return torch.cat(encoded_mean_frames, dim=-1), torch.cat(encoded_logvar_frames, dim=-1)

    def encode_frame(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        length = audio.size(-1)
        if length < self.chunk_length:
            audio = F.pad(audio, (0, self.chunk_length - length))
        mean, logvar = self.encoder(audio)
        return mean, logvar

    def decode(self,
               encoded: torch.Tensor) -> torch.Tensor:
        assert encoded.dim() == 3, ("Only 3-dim encoded latent representation is supported. Shape should be "
                                    "(batch_size, dim, num_frames * frame_size).")
        assert encoded.size(-1) % self.frame_size == 0, \
            f"Second dimension of encoded ({encoded.size(-1)}) must be divisible by frame_size ({self.frame_size})."
        num_frames = encoded.size(-1) // self.frame_size
        audio_list = []
        for i in range(num_frames):
            frame = encoded[:, :, i * self.frame_size:(i + 1) * self.frame_size]
            audio_list.append(self.decode_frame(frame))
        return linear_overlap_add(audio_list, self.chunk_stride)

    def decode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        audio = self.decoder(frame)
        return audio

    @staticmethod
    def _reparameterize(mean: torch.Tensor,
                        log_var: torch.Tensor,
                        eps: Optional[torch.Tensor] = None,
                        rank: int = 0,
                        world_size: int = 1) -> torch.Tensor:
        if eps is None:
            eps = torch.randn(world_size, *mean.shape)[rank].to(mean.device).to(mean.dtype)
        std = torch.exp(0.5 * log_var)
        y = mean + eps * std
        return y

    def remove_weight_norm_(self):
        self.encoder.remove_weight_norm_()
        self.decoder.remove_weight_norm_()

    def forward(self,
                audio_frame: torch.Tensor,
                eps: Optional[torch.Tensor] = None,
                rank: int = 0,
                world_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_var = self.encode_frame(audio_frame)
        z = self._reparameterize(mean, log_var, eps, rank, world_size)
        reconstructed = self.decode_frame(z)
        return reconstructed, mean, log_var