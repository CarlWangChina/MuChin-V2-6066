import torch
import torch.nn as nn
from typing import Optional
from einops import rearrange
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger
logger = get_logger(__name__)
from .fft import FFTBlocks
from .fs2_variance import FS2Variance

class FastSpeech2(nn.Module):
    def __init__(self,                 phoneme_vocab_size: int,                 hidden_size: int,                 num_encoder_layers: int,                 num_decoder_layers: int,                 kernel_size: int,                 num_heads: int,                 output_dim: int,                 duration_predictor_layers: int,                 duration_predictor_kernel_size: int,                 pitch_predictor_layers: int,                 pitch_predictor_kernel_size: int,                 energy_predictor_layers: int,                 energy_predictor_kernel_size: int,                 phoneme_padding_token_id: int,                 dropout: float = 0.1,                 num_speakers: int = 1,                 melody_feature_rate: int = 75,                 sampling_rate: int = 44100,                 vocoder_hop_size: int = 512,                 pitch_max: int = 128,                 norm_type: str = "layer_norm",                 padding_type: str = "same",                 act_type: str = "gelu",                 device: str = "cpu"):
        super(FastSpeech2, self).__init__()
        self.vocab_size = phoneme_vocab_size
        self.melody_feature_rate = melody_feature_rate
        self.mel_feature_rate = sampling_rate / vocoder_hop_size
        self.mel_melody_rate_ratio = self.mel_feature_rate / self.melody_feature_rate
        self.device = device
        self.phoneme_embedding = nn.Embedding(phoneme_vocab_size,                                              hidden_size,                                              padding_idx=phoneme_padding_token_id,                                              device=device)
        self.encoder = FFTBlocks(hidden_size=hidden_size,                                 num_layers=num_encoder_layers,                                 kernel_size=kernel_size,                                 num_heads=num_heads,                                 dropout=dropout,                                 norm_type=norm_type,                                 padding_type=padding_type,                                 act_type=act_type,                                 device=device)
        self.decoder = FFTBlocks(hidden_size=hidden_size,                                 num_layers=num_decoder_layers,                                 kernel_size=kernel_size,                                 num_heads=num_heads,                                 dropout=dropout,                                 norm_type=norm_type,                                 padding_type=padding_type,                                 act_type=act_type,                                 device=device)
        self.mel_out_proj = nn.Linear(hidden_size, output_dim, bias=True, device=device)
        if num_speakers > 1:
            self.speakers_embedding = nn.Embedding(num_speakers, hidden_size, device=device)
        else:
            self.speakers_embedding = None
        self.variance_adaptor = FS2Variance(hidden_size=hidden_size,                                            duration_predictor_layers=duration_predictor_layers,                                            duration_predictor_kernel_size=duration_predictor_kernel_size,                                            pitch_predictor_layers=pitch_predictor_layers,                                            pitch_predictor_kernel_size=pitch_predictor_kernel_size,                                            energy_predictor_layers=energy_predictor_layers,                                            energy_predictor_kernel_size=energy_predictor_kernel_size,                                            dropout=dropout,                                            padding_type=padding_type,                                            pitch_max=pitch_max,                                            rate_ratio=self.mel_melody_rate_ratio,                                            device=device)

    def forward(self,                phoneme_tokens: torch.Tensor,                phoneme_duration: Optional[torch.Tensor] = None,                pitch_tokens: Optional[torch.Tensor] = None,                speaker_ids: Optional[torch.Tensor] = None,                mask: Optional[torch.Tensor] = None) -> (torch.Tensor, torch.Tensor):
        assert phoneme_tokens.dim() == 2, "The phoneme tokens should be 2-dimensional."
        assert phoneme_tokens.dtype == torch.long, "The phoneme tokens should be of the torch.long dtype."
        if phoneme_duration is not None:
            assert phoneme_duration.shape == phoneme_tokens.shape, \                "The phoneme duration should have the same shape as the phoneme tokens."
        if speaker_ids is not None:
            assert speaker_ids.shape == phoneme_tokens.shape, \                "The speaker IDs should have the same shape as the phoneme tokens."
        if pitch_tokens is not None:
            assert pitch_tokens.dim() == 2, "The pitch tokens should be 2-dimensional."
            assert pitch_tokens.dtype == torch.long, "The pitch tokens should be of the torch.long dtype."
            assert pitch_tokens.shape[0] == phoneme_tokens.shape[0], \                "The batch dimension of the pitch tokens should be the same as the phoneme tokens."
        if mask is not None:
            assert mask.shape == phoneme_tokens.shape, \                "The mask should have the same shape as the phoneme tokens."
            phoneme_h_mask = rearrange(mask.float(), "b t -> b t ()")
            phoneme_h_mask = phoneme_h_mask.expand(-1, -1, phoneme_tokens.shape[-1])
        else:
            phoneme_h_mask = None
        h = self.phoneme_embedding(phoneme_tokens)
        h = self.encoder(h, padding_mask=mask)
        if speaker_ids is not None:
            assert self.speakers_embedding is not None, \                "The speaker embedding layer is not initialized because the hyper param num_speakers = 1."
            h = h + self.speakers_embedding(speaker_ids)
            if phoneme_h_mask is not None:
                h *= phoneme_h_mask
        h, mask = self.variance_adaptor(phoneme=h,                                        phoneme_duration=phoneme_duration,                                        pitch_tokens=pitch_tokens,                                        speaker_ids=speaker_ids,                                        mask=mask)
        decoder_inp = h
        h = self.decoder(h, padding_mask=mask)
        h = self.mel_out_proj(h)
        return h, decoder_inp