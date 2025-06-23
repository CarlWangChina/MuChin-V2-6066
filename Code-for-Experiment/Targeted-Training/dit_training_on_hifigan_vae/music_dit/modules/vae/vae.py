import Code_for_Experiment.Metrics.music_understanding_model.tensorboardX.tests.test_numpy
import torch
import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.amp.rnn_compat
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from einops import rearrange
import numpy as np
from music_dit.utils import get_logger, get_hparams, linear_overlap_add
from .seanet import SEANetEncoder, SEANetDecoder

logger = get_logger(__name__)

class VAEModel(nn.Module):
    def __init__(self):
        super(VAEModel, self).__init__()
        hparams = get_hparams()
        device = hparams.device
        self.encoder = SEANetEncoder(channels=hparams.vae.num_channels,
                                     dimension=hparams.vae.embedding_dim,
                                     n_filters=hparams.vae.num_filters,
                                     n_residual_layers=hparams.vae.num_residual_layers,
                                     ratios=hparams.vae.upsampling_ratios,
                                     activation='ELU',
                                     norm=hparams.vae.norm_type,
                                     kernel_size=hparams.vae.kernel_size,
                                     last_kernel_size=hparams.vae.last_kernel_size,
                                     residual_kernel_size=hparams.vae.residual_kernel_size,
                                     causal=hparams.vae.causal,
                                     pad_mode=hparams.vae.pad_mode,
                                     compress=hparams.vae.compress,
                                     lstm=hparams.vae.num_lstm_layers).to(device)
        self.decoder = SEANetDecoder(channels=hparams.vae.num_channels,
                                     dimension=hparams.vae.embedding_dim,
                                     n_filters=hparams.vae.num_filters,
                                     n_residual_layers=hparams.vae.num_residual_layers,
                                     ratios=hparams.vae.upsampling_ratios,
                                     activation='ELU',
                                     norm=hparams.vae.norm_type,
                                     kernel_size=hparams.vae.kernel_size,
                                     last_kernel_size=hparams.vae.last_kernel_size,
                                     residual_kernel_size=hparams.vae.residual_kernel_size,
                                     causal=hparams.vae.causal,
                                     pad_mode=hparams.vae.pad_mode,
                                     compress=hparams.vae.compress,
                                     lstm=hparams.vae.num_lstm_layers,
                                     trim_right_ratio=hparams.vae.trim_right_ratio).to(device)
        self.sampling_rate = hparams.vae.sampling_rate
        self.num_channels = hparams.vae.num_channels
        self.embedding_dim = hparams.vae.embedding_dim
        self.chunk_length = hparams.vae.chunk_length
        self.chunk_stride = hparams.vae.chunk_stride
        ratio_prod = int(np.prod(hparams.vae.upsampling_ratios))
        assert self.chunk_length % ratio_prod == 0, \
            f"chunk_length ({self.chunk_length}) must be divisible by the product of upsampling ratios ({ratio_prod})"
        self.frame_size = self.chunk_length // ratio_prod
        self.time_unit = self.chunk_stride / self.sampling_rate
        self.frame_rate = self.sampling_rate / self.chunk_stride

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self,
               audio: torch.Tensor,
               padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        encoded_frames = []
        for offset in range(0, seq_length, self.chunk_stride):
            audio_chunk = audio[:, :, offset:offset + self.chunk_length]
            encoded_frames.append(self._encode_frame(audio_chunk).unsqueeze(1))
        return torch.cat(encoded_frames, dim=1)

    def _encode_frame(self, audio: torch.Tensor) -> torch.Tensor:
        length = audio.size(-1)
        if length < self.chunk_length:
            audio = F.pad(audio, (0, self.chunk_length - length))
        emb = self.encoder(audio)
        return emb

    def decode(self,
               encoded: torch.Tensor) -> torch.Tensor:
        assert encoded.dim() == 4, ("Only 4-dim encoded latent representation is supported. Shape should be "
                                    "(batch_size, num_frames, frame_size, embedding_dim).")
        audio_list = []
        encoded = rearrange(encoded, 'b f ... -> f b ...').contiguous()
        for frame in encoded:
            audio_list.append(self._decode_frame(frame))
        return linear_overlap_add(audio_list, self.chunk_stride)

    def _decode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        audio = self.decoder(frame)
        return audio