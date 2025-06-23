import math
import numpy as np
from pathlib import Path
import typing as tp
import torch
from torch import nn
from . import quantization as qt
from . import modules as m
from .utils import _check_checksum, _linear_overlap_add, _get_checkpoint_url

ROOT_URL = "https://dl.fbaipublicfiles.com/encodec/v0/"

EncodedFrame = tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]

class LMModel(nn.Module):
    def __init__(self, n_q: int = 32, card: int = 1024, dim: int = 200, **kwargs):
        super().__init__()
        self.card = card
        self.n_q = n_q
        self.dim = dim
        self.transformer = m.StreamingTransformerEncoder(dim=dim, **kwargs)
        self.emb = nn.ModuleList([nn.Embedding(card + 1, dim) for _ in range(n_q)])
        self.linears = nn.ModuleList([nn.Linear(dim, card) for _ in range(n_q)])

    def forward(
        self,
        indices: torch.Tensor,
        states: tp.Optional[tp.List[torch.Tensor]] = None,
        offset: int = 0,
    ):
        B, K, T = indices.shape
        input_ = sum([self.emb[k](indices[:, k]) for k in range(K)])
        out, states, offset = self.transformer(input_, states, offset)
        logits = torch.stack([self.linears[k](out) for k in range(K)], dim=1).permute(
            0, 3, 1, 2
        )
        return torch.softmax(logits, dim=1), states, offset

class EncodecModel(nn.Module):
    def __init__(
        self,
        encoder: m.SEANetEncoder,
        decoder: m.SEANetDecoder,
        quantizer: qt.ResidualVectorQuantizer,
        target_bandwidths: tp.List[float],
        sample_rate: int,
        channels: int,
        normalize: bool = False,
        segment: tp.Optional[float] = None,
        overlap: float = 0.01,
        name: str = "unset",
    ):
        super().__init__()
        self.bandwidth: tp.Optional[float] = None
        self.target_bandwidths = target_bandwidths
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
        self.sample_rate = sample_rate
        self.channels = channels
        self.normalize = normalize
        self.segment = segment
        self.overlap = overlap
        self.frame_rate = math.ceil(self.sample_rate / np.prod(self.encoder.ratios))
        self.name = name
        self.bits_per_codebook = int(math.log2(self.quantizer.bins))
        assert (
            2**self.bits_per_codebook == self.quantizer.bins
        ), "quantizer bins must be a power of 2."

    @property
    def segment_length(self) -> tp.Optional[int]:
        if self.segment is None:
            return None
        return int(self.segment * self.sample_rate)

    @property
    def segment_stride(self) -> tp.Optional[int]:
        segment_length = self.segment_length
        if segment_length is None:
            return None
        return max(1, int((1 - self.overlap) * segment_length))

    def encode(self, x: torch.Tensor) -> tp.List[EncodedFrame]:
        assert x.dim() == 3
        _, channels, length = x.shape
        assert channels > 0 and channels <= 2
        segment_length = self.segment_length
        if segment_length is None:
            segment_length = length
            stride = length
        else:
            stride = self.segment_stride
            assert stride is not None
        encoded_frames: tp.List[EncodedFrame] = []
        for offset in range(0, length, stride):
            frame = x[:, :, offset : offset + segment_length]
            encoded_frames.append(self._encode_frame(frame))
        return encoded_frames

    def _encode_frame(self, x: torch.Tensor) -> EncodedFrame:
        length = x.shape[-1]
        duration = length / self.sample_rate
        assert self.segment is None or duration <= 1e-5 + self.segment
        if self.normalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None
        emb = self.encoder(x)
        codes = self.quantizer.encode(emb, self.frame_rate, self.bandwidth)
        codes = codes.transpose(0, 1)
        return codes, scale

    def decode(self, encoded_frames: tp.List[EncodedFrame]) -> torch.Tensor:
        segment_length = self.segment_length
        if segment_length is None:
            assert len(encoded_frames) == 1
            return self._decode_frame(encoded_frames[0])
        frames = [self._decode_frame(frame) for frame in encoded_frames]
        return _linear_overlap_add(frames, self.segment_stride or 1)

    def _decode_frame(self, encoded_frame: EncodedFrame) -> torch.Tensor:
        codes, scale = encoded_frame
        codes = codes.transpose(0, 1)
        emb = self.quantizer.decode(codes)
        out = self.decoder(emb)
        if scale is not None:
            out = out * scale.view(-1, 1, 1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frames = self.encode(x)
        return self.decode(frames)[:, :, : x.shape[-1]]

    def set_target_bandwidth(self, bandwidth: float):
        if bandwidth not in self.target_bandwidths:
            raise ValueError(
                f"This model doesn't support the bandwidth {bandwidth}. "
                f"Select one of {self.target_bandwidths}."
            )
        self.bandwidth = bandwidth

    def get_lm_model(self) -> LMModel:
        device = next(self.parameters()).device
        lm = LMModel(
            self.quantizer.n_q,
            self.quantizer.bins,
            num_layers=5,
            dim=200,
            past_context=int(3.5 * self.frame_rate),
        ).to(device)
        checkpoints = {
            "encodec_24khz": "encodec_lm_24khz-1608e3c0.th",
            "encodec_48khz": "encodec_lm_48khz-7add9fc3.th",
        }
        try:
            checkpoint_name = checkpoints[self.name]
        except KeyError:
            raise RuntimeError("No LM pre-trained for the current Encodec model.")
        url = _get_checkpoint_url(ROOT_URL, checkpoint_name)
        print("-------")
        print(url)
        print("-------")
        state = torch.hub.load_state_dict_from_url(
            url, map_location="cpu", check_hash=True
        )
        lm.load_state_dict(state)
        lm.eval()
        return lm

    @staticmethod
    def _get_model(
        target_bandwidths: tp.List[float],
        sample_rate: int = 24_000,
        channels: int = 1,
        causal: bool = True,
        model_norm: str = "weight_norm",
        audio_normalize: bool = False,
        segment: tp.Optional[float] = None,
        name: str = "unset",
    ):
        encoder = m.SEANetEncoder(channels=channels, norm=model_norm, causal=causal)
        decoder = m.SEANetDecoder(channels=channels, norm=model_norm, causal=causal)
        n_q = int(
            1000
            * target_bandwidths[-1]
            // (math.ceil(sample_rate / encoder.hop_length) * 10)
        )
        quantizer = qt.ResidualVectorQuantizer(
            dimension=encoder.dimension,
            n_q=n_q,
            bins=1024,
        )
        model = EncodecModel(
            encoder,
            decoder,
            quantizer,
            target_bandwidths,
            sample_rate,
            channels,
            normalize=audio_normalize,
            segment=segment,
            name=name,
        )
        return model

    @staticmethod
    def _get_pretrained(checkpoint_name: str, repository: tp.Optional[Path] = None):
        if repository is not None:
            if not repository.is_dir():
                raise ValueError(f"{repository} must exist and be a directory.")
            file = repository / checkpoint_name
            checksum = file.stem.split("-")[1]
            _check_checksum(file, checksum)
            return torch.load(file)
        else:
            url = _get_checkpoint_url(ROOT_URL, checkpoint_name)
            return torch.hub.load_state_dict_from_url(
                url, map_location="cpu", check_hash=True
            )

    @staticmethod
    def encodec_model_24khz(
        pretrained: bool = True, repository: tp.Optional[Path] = None
    ):
        if repository:
            assert pretrained
        target_bandwidths = [1.5, 3.0, 6, 12.0, 24.0]
        checkpoint_name = "encodec_24khz-d7cc33bc.th"
        sample_rate = 24_000
        channels = 1
        model = EncodecModel._get_model(
            target_bandwidths,
            sample_rate,
            channels,
            causal=True,
            model_norm="weight_norm",
            audio_normalize=False,
            name="encodec_24khz" if pretrained else "unset",
        )
        if pretrained:
            state_dict = EncodecModel._get_pretrained(checkpoint_name, repository)
            model.load_state_dict(state_dict)
        model.eval()
        return model

    @staticmethod
    def encodec_model_48khz(
        pretrained: bool = True, repository: tp.Optional[Path] = None
    ):
        if repository:
            assert pretrained
        target_bandwidths = [3.0, 6.0, 12.0, 24.0]
        checkpoint_name = "encodec_48khz-7e698e3e.th"
        sample_rate = 48_000
        channels = 2
        model = EncodecModel._get_model(
            target_bandwidths,
            sample_rate,
            channels,
            causal=False,
            model_norm="time_group_norm",
            audio_normalize=True,
            segment=1.0,
            name="encodec_48khz" if pretrained else "unset",
        )
        if pretrained:
            state_dict = EncodecModel._get_pretrained(checkpoint_name, repository)
            model.load_state_dict(state_dict)
        model.eval()
        return model

def test():
    from itertools import product
    import torchaudio
    bandwidths = [3, 6, 12, 24]
    models = {
        "encodec_24khz": EncodecModel.encodec_model_24khz,
        "encodec_48khz": EncodecModel.encodec_model_48khz,
    }
    for model_name, bw in product(models.keys(), bandwidths):
        model = models[model_name]()
        model.set_target_bandwidth(bw)
        audio_suffix = model_name.split("_")[1][:3]
        wav, sr = torchaudio.load(f"test_{audio_suffix}.wav")
        wav = wav[:, : model.sample_rate * 2]
        wav_in = wav.unsqueeze(0)
        wav_dec = model(wav_in)[0]
        assert wav.shape == wav_dec.shape, (wav.shape, wav_dec.shape)

if __name__ == "__main__":
    test()