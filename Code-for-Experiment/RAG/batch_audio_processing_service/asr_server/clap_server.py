import torch
import torch.nn.functional as F
import torchaudio
import csv
import argparse
import laion_clap
from pathlib import Path
from tqdm.auto import tqdm
from ray import serve
import logging
from starlette.requests import Request
from starlette.responses import JSONResponse
from ama_prof_divi_codec import ama_prof_diviCodec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
CHECKPOINT_FILE = Path(__file__).parent.parent / "data" / "checkpoints" / "music_audioset_epoch_15_esc_90.14.pt"

def fetch_array(array, step, size):
    if step <= 0 or size <= 0:
        raise ValueError("step和size必须为正数")
    end_index = array.shape[0] - size + 1
    start_index = 0
    while start_index < end_index:
        yield array[start_index:start_index + size]
        start_index += step

@serve.deployment(name="CLAP", num_replicas=4, ray_actor_options={"num_cpus": 2, "num_gpus": 1})
class CLAPProcessor:
    def __init__(self, device="cuda"):
        self.clap_model = laion_clap.CLAP_Module(enable_fusion=False,
                                                amodel="HTSAT-base",
                                                tmodel="roberta").to(device)
        self.clap_model.load_ckpt(str(CHECKPOINT_FILE))
        self.clap_model.eval()
        self.sample_rate = self.clap_model.model_cfg["audio_cfg"]["sample_rate"]
        self.sample_skip = int(61440 * self.sample_rate / 24000)
        self.chunk_length = int(self.sample_rate * 10)
        self.clap_dim = self.clap_model.model_cfg["text_cfg"]["width"]
        self.device = device
        self.codec = ama_prof_diviCodec(num_workers=4)

    def resample_audio(self, waveform, orig_sample_rate):
        if orig_sample_rate != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_sample_rate, self.sample_rate)(waveform)
        res = waveform.mean(dim=0).numpy()
        return res

    @torch.no_grad()
    def get_clap_vector(self, audio: torch.Tensor) -> torch.Tensor:
        audio = audio.to(self.device)
        clap_tensor = []
        for chunk in fetch_array(audio, self.sample_skip, self.sample_rate * 10):
            chunk = chunk.view(1, -1)
            if chunk.size(-1) < self.chunk_length:
                chunk = F.pad(chunk, (0, self.chunk_length - chunk.size(-1)))
            clap = self.clap_model.get_audio_embedding_from_data(chunk, use_tensor=True)
            assert clap.size() == (1, self.clap_dim)
            clap_tensor.append(clap)
        return torch.cat(clap_tensor, dim=0).to(self.device)

    @torch.inference_mode()
    def process(self, audio_data, sr):
        audio = self.resample_audio(audio_data, sr)
        result = self.get_clap_vector(torch.from_numpy(audio).cuda())
        return result

    async def __call__(self, request: Request):
        audio_bytes = await request.body()
        if len(audio_bytes) == 0:
            return b""
        meta, data = self.codec.decode_bytes([audio_bytes])
        audio = data[0]
        sr = meta[0]['compressed']['sample_rate']
        result = self.process(audio, sr).view(-1).cpu().to(torch.float32).numpy().tobytes()
        return result

clap_app = CLAPProcessor.options(route_prefix="/CLAP").bind()