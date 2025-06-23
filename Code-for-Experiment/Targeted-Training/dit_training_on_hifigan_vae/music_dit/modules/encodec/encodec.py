import torch
import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.amp.rnn_compat
from typing import Optional, List
from einops import rearrange
from transformers import EncodecModel, EncodecFeatureExtractor
from music_dit.utils import get_hparams, get_audio_utils, get_logger, linear_overlap_add
import nn

logger = get_logger(__name__)

class EncodecVAE(nn.Module):
    def __init__(self, 
                 batch_size: Optional[int] = None):
        super().__init__()
        self.hparams = get_hparams()
        device = self.hparams.device
        self.audio_utils = get_audio_utils()
        self.model = EncodecModel.from_pretrained(self.hparams.encodec.pretrained_model).to(device)
        self.processor = EncodecFeatureExtractor.from_pretrained(self.hparams.encodec.pretrained_model)
        self.batch_size = batch_size if batch_size is not None else self.hparams.encodec.batch_size
        self.normalize_loudness = self.hparams.encodec.normalize_loudness
        self.input_loudness = self.hparams.encodec.input_loudness
        self.output_loudness = self.hparams.encodec.output_loudness

    @property
    def model_name(self) -> str:
        return self.hparams.encodec.pretrained_model

    @property
    def num_channels(self) -> int:
        return self.model.config.audio_channels

    @property
    def sampling_rate(self) -> int:
        return self.model.config.sampling_rate

    @property
    def segment_length(self) -> int:
        return self.model.config.chunk_length

    @property
    def segment_stride(self) -> int:
        return self.model.config.chunk_stride

    @property
    def frame_rate(self) -> int:
        return self.model.config.frame_rate

    @property
    def embedding_dim(self) -> int:
        return self.model.config.codebook_dim

    @property
    def device(self):
        return self.model.device

    @torch.no_grad()
    def _encode_frame(self, 
                      audio: torch.Tensor, 
                      padding_mask: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        audio = audio * padding_mask
        scale = torch.ones(audio.size(0), device=self.device)
        if self.model.config.normalize:
            mono = torch.sum(audio, 1, keepdim=True) / audio.size(1)
            scale = mono.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8
            audio = audio / scale
        embeddings = self.model.encoder(audio)
        embeddings = rearrange(embeddings, 'b d s -> b s d').contiguous()
        return embeddings, scale

    @torch.no_grad()
    def encode(self, 
               audio: torch.Tensor, 
               sampling_rate: int) -> (torch.Tensor, torch.Tensor):
        assert audio.dim() == 2, "The audio data must be 2D tensor, in (num_channels, num_samples)."
        if sampling_rate != self.sampling_rate:
            audio = self.audio_utils.resample(audio, 
                                              sampling_rate, 
                                              self.sampling_rate, 
                                              self.num_channels)
        if self.normalize_loudness:
            audio = self.audio_utils.normalize_loudness(audio, 
                                                        self.sampling_rate, 
                                                        self.input_loudness)
        inp = self.processor(audio, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True)
        input_values = inp["input_values"]
        padding_mask = inp["padding_mask"]
        if padding_mask is None:
            padding_mask = torch.ones_like(input_values).bool()
        encoded_frames = []
        scales = []
        chunk_length = self.segment_length
        stride = self.segment_stride
        input_length = input_values.size(-1)
        step = chunk_length - stride
        if (input_length % stride) - step != 0:
            raise ValueError(
                "The input length is not properly padded for batched chunked decoding. Make sure to pad the input "
                "correctly."
            )
        for offset in range(0, input_length - step, stride):
            mask = padding_mask[..., offset: offset + chunk_length].bool().to(self.device)
            frame = input_values[:, :, offset: offset + chunk_length].to(self.device)
            encoded_frame, scale = self._encode_frame(frame, mask)
            encoded_frames.append(encoded_frame[0].cpu())
            scales.append(scale[0].cpu())
        encoded_frames = torch.stack(encoded_frames)
        scales = torch.cat(scales).squeeze(-1)
        return encoded_frames, scales

    def _decode_frame(self, embeddings: torch.Tensor, 
                      scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        embeddings = rearrange(embeddings, 'b s d -> b d s').contiguous()
        outputs = self.model.decoder(embeddings)
        if scale is not None:
            outputs = outputs * scale.view(-1, 1, 1)
        return outputs

    @torch.no_grad()
    def decode(self, 
               encoded: torch.Tensor, 
               scales: Optional[torch.Tensor]) -> torch.Tensor:
        assert encoded.dim() == 3, "The encoded data must be 3D tensor, in (seq_len, frame_size, embedding_dim)."
        assert encoded.size(1) == self.frame_rate, f"The frame rate must be equal to {self.frame_rate}."
        if scales is not None:
            assert scales.dim() == 1, "The scales mask must be 1D tensor."
        stride = self.segment_stride
        decoded_frames = []
        for frame, scale in zip(encoded, scales):
            frame = frame.unsqueeze(0)
            frames = self._decode_frame(frame.to(self.device), scale.to(self.device))
            decoded_frames.append(frames.cpu())
        audio_values = linear_overlap_add(decoded_frames, stride or 1).squeeze(0)
        if self.normalize_loudness:
            audio_values = self.audio_utils.normalize_loudness(audio_values, 
                                                               self.sampling_rate, 
                                                               self.output_loudness)
        return audio_values