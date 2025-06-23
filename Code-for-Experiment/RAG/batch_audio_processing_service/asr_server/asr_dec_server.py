import whisperx
import gc
import torch
import json
import ray
import torchaudio
import io
import re
import string
from Code_for_Experiment.Metrics.music_understanding_model.tensorboardX.tests.test_numpy import *
from pathlib import Path
from ray import serve
import logging
from starlette.requests import Request
from starlette.responses import JSONResponse
from typing import Optional
import opencc
from ama_prof_divi_codec import ama_prof_diviCodec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
matcher = re.compile(r"Text:(.*?) Timestamps:\((.*), (.*)\)")
filter_pattern = re.compile(r"^[0-9\s\-{re.escape(string.punctuation)}]*$")
hallucinations = []
hallucination_table_path = Path(__file__).parent.parent.joinpath('data').joinpath('hallucinations.txt')
with open(hallucination_table_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line == "" or line.startswith("#"):
            continue
        hallucinations.append(line)

def _filter_lyrics_text(text: str) -> str:
    text = re.sub(r'\(.*?\)', '', text)
    for hallucination in hallucinations:
        if hallucination.lower() in text:
            return ""
    if filter_pattern.match(text):
        return ""
    text = text.strip()
    if len(text) == 0:
        return ""
    return text

@serve.deployment(name="ASRDec", num_replicas=4, ray_actor_options={"num_cpus": 2, "num_gpus": 1})
class ASRDec:
    def __init__(self, device="cuda", device_index=0):
        self.device = device
        self.device_index = device_index
        self.batch_size = 16
        self.compute_type = "float16"
        self.sample_rate = 16000
        self.model = whisperx.load_model("large-v2", self.device, compute_type=self.compute_type, device_index=self.device_index)
        self.align_model_cache = {}
        self.opencc_converter = opencc.OpenCC("t2s")
        self.codec = ama_prof_diviCodec(num_workers=4)

    def get_align_model(self, lang):
        if lang in self.align_model_cache:
            return self.align_model_cache[lang]
        else:
            model_a, metadata = whisperx.load_align_model(language_code=lang, device=self.device)
            self.align_model_cache[lang] = (model_a, metadata)
            return (model_a, metadata)

    def resample_audio(self, waveform, orig_sample_rate):
        if orig_sample_rate != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_sample_rate, self.sample_rate)(waveform)
        res = waveform.mean(dim=0).numpy()
        return res

    @torch.inference_mode()
    def process(self, audio_data, sr):
        audio = self.resample_audio(audio_data, sr)
        result = self.model.transcribe(audio, batch_size=self.batch_size)
        model_a, metadata = self.get_align_model(result["language"])
        result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)
        return result

    async def __call__(self, request: Request):
        audio_bytes = await request.body()
        if len(audio_bytes) == 0:
            return ""
        meta, data = self.codec.decode_bytes([audio_bytes])
        audio = data[0]
        sr = meta[0]['compressed']['sample_rate']
        result = self.process(audio, sr)
        res_str = ""
        for segment in result["segments"]:
            line = _filter_lyrics_text(self.opencc_converter.convert(segment["text"]).lower())
            if len(line) > 0:
                res_str += "[" + str(segment["start"]) + "," + str(segment["end"]) + "]" + line + "\n"
        return JSONResponse(res_str)

asr_app = ASRDec.options(route_prefix="/ASR").bind()