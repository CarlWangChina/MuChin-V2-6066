import torch
import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.amp.rnn
from typing import Optional
from einops import rearrange
from tqdm import tqdm
from transformers import EncodecModel, EncodecFeatureExtractor

class EncodecModelWrapper(torch.nn.Module):
    def __init__(self, hparams: dict):
        super(EncodecModelWrapper, self).__init__()
        self.hparams = hparams
        self.enc_hparams = hparams["ama-prof-divi"]["models"]["acoustic"]["encodec"]
        device = hparams["ama-prof-divi"]["device"]
        self.model = EncodecModel.from_pretrained(self.enc_hparams["pretrained_model"]).to(device)
        self.processor = EncodecFeatureExtractor.from_pretrained(self.enc_hparams["pretrained_model"])
        self.bandwidth = self.enc_hparams["bandwidth"]
        assert self.bandwidth in self.model.config.target_bandwidths, \
            f"bandwidth {self.bandwidth} is not valid.  Should be one of {self.model.config.target_bandwidths}"

    @property
    def model_name(self) -> str:
        return self.enc_hparams["pretrained_model"]

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

    @property
    def device(self):
        return self.model.device

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
        result = self.model.decode(audio_codes, audio_scales=audio_scales)["audio_values"]
        assert result.dim() == 3
        assert result.shape[0] == audio_codes.shape[1]
        assert result.shape[1] == self.num_channels
        return result

    @torch.no_grad()
    def decode_whole(self, audio_codes: torch.IntTensor, audio_scales: Optional[torch.Tensor] = None, n_chunks: int = 2) -> [torch.Tensor]:
        assert audio_codes.dim() == 4, \
            f"audio_codes should have 4 dimensions, but got {audio_codes.dim()}"
        if audio_scales is not None:
            if audio_scales.dim() == 1:
                audio_scales = audio_scales.unsqueeze(-1)
            assert audio_scales.dim() == 2, \
                f"audio_scales should have 2 dimensions, but got {audio_scales.dim()}"
            assert audio_codes.shape[0] == audio_scales.shape[0], \
                f"audio_codes.shape[0] = {audio_codes.shape[0]}, audio_scales.shape[0] = {audio_scales.shape[0]}"
        stride = self.segment_stride
        audio = torch.zeros(1, 2, stride * audio_codes.shape[0], device="cpu")
        for i in tqdm(range(0, audio_codes.shape[0], n_chunks)):
            audio_codes_window = audio_codes[i:i + n_chunks, ...].to(self.device)
            audio_scales_window = audio_scales[i:i + n_chunks, ...].to(self.device) if audio_scales is not None else None
            audio_window = self.decode(audio_codes_window, audio_scales_window)["audio_values"].cpu()
            win_length = min(audio_window.shape[1], stride * n_chunks, audio.shape[1] - stride * i)
            audio[:, :, stride * i:stride * i + win_length] = audio_window[:, :, :win_length]
        return audio