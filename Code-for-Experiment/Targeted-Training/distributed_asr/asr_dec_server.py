import whisperx
import gc
import torch
import json
import ray
import torchaudio
import io
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 4, "num_gpus": 1})
class ASRDec:
    def __init__(self, device="cuda", device_index=0):
        self.device = device
        self.device_index = device_index
        self.batch_size = 16
        self.compute_type = "float16"
        self.sample_rate = 16000
        self.model = whisperx.load_model("large-v2", self.device, compute_type=self.compute_type, device_index=self.device_index)
        self.align_model_cache = {}

    def get_align_model(self, lang):
        if lang in self.align_model_cache:
            return self.align_model_cache[lang]
        else:
            model_a, metadata = whisperx.load_align_model(language_code=lang, device=self.device)
            self.align_model_cache[lang] = (model_a, metadata)
            return (model_a, metadata)

    def resample_audio(self, audio_bytes, orig_sample_rate):
        waveform, _ = torchaudio.load(io.BytesIO(audio_bytes), normalize=True)
        if orig_sample_rate != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_sample_rate, self.sample_rate)(waveform)
        return waveform.squeeze().numpy()

    @torch.inference_mode()
    def process(self, audio_bytes):
        orig_waveform, orig_sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
        audio = self.resample_audio(audio_bytes, orig_sample_rate)
        result = self.model.transcribe(audio, batch_size=self.batch_size)
        model_a, metadata = self.get_align_model(result["language"])
        result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)
        return result

    async def __call__(self, request: Request):
        audio_bytes = await request.body()
        result = self.process(audio_bytes)
        return JSONResponse(result)

asr_app = ASRDec.bind()