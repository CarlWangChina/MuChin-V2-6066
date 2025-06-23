import torch
import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.amp.rnn_compat
from typing import Optional
from einops import rearrange
from transformers import EncodecModel, EncodecFeatureExtractor
import torch.nn as nn

class EncodecModelWrapper(nn.Module):
    def __init__(self, device: str):
        super(EncodecModelWrapper, self).__init__()
        self.device = device
        self.model = EncodecModel.from_pretrained("facebook/encodec_48khz").to(self.device)
        self.processor = EncodecFeatureExtractor.from_pretrained("facebook/encodec_48khz")
        self.bandwidth = 12.0
        assert self.bandwidth in self.model.config.target_bandwidths, \
            f"bandwidth {self.bandwidth} is not valid.  Should be one of {self.model.config.target_bandwidths}"

    @property
    def num_channels(self) -> int:
        return self.model.config.audio_channels

    @property
    def sampling_rate(self) -> int:
        return self.model.config.sampling_rate

    @property
    def num_quantizers(self) -> int:
        return self.model.quantizer.get_num_quantizers_for_bandwidth(self.bandwidth)

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
    def codebook_size(self) -> int:
        return self.model.quantizer.codebook_size

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    @torch.no_grad()
    def encode(self, audio: [torch.Tensor]) -> dict:
        if not type(audio) is list:
            assert torch.is_tensor(audio), \
                f"audio should be either a tensor or a list of tensors, but got {type(audio)}"
            audio = [audio]
        num_batches = len(audio)
        if num_batches == 0:
            return {
                "audio_codes": torch.empty((0, 0, 0, 0), dtype=torch.long, device=self.device),
                "audio_scales": torch.empty((0, 0), dtype=torch.float32, device=self.device)
            }
        inputs = []
        max_num_samples = 0
        for t in audio:
            assert t.dim() == 2, \
                f"audio tensor should have 2 dimensions, but got {t.dim()}"
            assert t.shape[0] == self.num_channels, \
                f"audio tensor should have {self.num_channels} channels, but got {t.shape[0]}"
            inp = self.processor(t.cpu(),
                                 sampling_rate=self.sampling_rate,
                                 return_tensors="pt",
                                 padding=True)
            assert inp["input_values"].dim() == 3
            assert inp["input_values"].shape[0] == 1
            max_num_samples = max(max_num_samples, inp["input_values"].shape[2])
            inputs.append(inp)
        input_values = torch.zeros((num_batches, self.num_channels, max_num_samples),
                                   dtype=torch.float32,
                                   device=self.device)
        padding_mask = torch.zeros((num_batches, max_num_samples),
                                   dtype=torch.bool,
                                   device=self.device)
        for i, inp in enumerate(inputs):
            input_values[i, :, :inp["input_values"].shape[2]] = inp["input_values"]
            padding_mask[i, :inp["padding_mask"].shape[1]] = inp["padding_mask"]
        padding_mask = rearrange(padding_mask, "b l -> b () l")
        encode_result = self.model.encode(input_values,
                                          padding_mask=padding_mask,
                                          return_dict=True,
                                          bandwidth=self.bandwidth)
        audio_codes = encode_result["audio_codes"]
        assert audio_codes.dim() == 4
        assert audio_codes.shape[1] == num_batches
        assert audio_codes.shape[2] == self.num_quantizers, \
            f"audio_codes.shape[2] = {audio_codes.shape[2]}, num_quantizers = {self.num_quantizers}"
        audio_scales = torch.stack(encode_result["audio_scales"])
        audio_scales = rearrange(audio_scales, "s b () () -> s b")
        assert audio_scales.dim() == 2
        assert audio_scales.shape[0] == audio_codes.shape[0]
        assert audio_scales.shape[1] == num_batches
        return {
            "audio_codes": audio_codes,
            "audio_scales": audio_scales
        }

    @torch.no_grad()
    def decode(self, audio_codes: torch.IntTensor, audio_scales: Optional[torch.Tensor] = None) -> [torch.Tensor]:
        assert audio_codes.dim() == 4
        if audio_scales is not None:
            assert audio_scales.dim() == 2
            assert audio_codes.shape[0] == audio_scales.shape[0]
            assert audio_codes.shape[1] == audio_scales.shape[1]
        else:
            audio_scales = torch.ones((audio_codes.shape[0], audio_codes.shape[1]),
                                      dtype=torch.float32,
                                      device=self.device)
        assert audio_codes.shape[2] == self.num_quantizers
        result = self.model.decode(audio_codes,
                                   audio_scales=audio_scales)["audio_values"]
        assert result.dim() == 3
        assert result.shape[0] == audio_codes.shape[1]
        assert result.shape[1] == self.num_channels
        return result