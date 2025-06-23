import torch
import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.amp.rnn_compat
import torchaudio
import pyloudnorm as pyln
from einops import rearrange
import demucs.pretrained
import demucs.audio
import demucs.apply
from music_dit.utils import get_hparams, get_logger

logger = get_logger(__name__)

class AudioUtils(torch.nn.Module):
    def __init__(self):
        super(AudioUtils, self).__init__()
        self.hparams = get_hparams()
        device = self.hparams.device
        self.demucs = demucs.pretrained.get_model(self.hparams.demucs.pretrained_model).to(device)
        logger.info("Loaded pretrained demucs model: %s", self.hparams.demucs.pretrained_model)
        self.demucs_stems_dict = self._get_demucs_stems_dict()
        logger.info("Demucs stems: %s", self.demucs_stems_dict)
        self.demucs.eval()
        for param in self.demucs.parameters():
            param.requires_grad = False

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def demucs_sampling_rate(self):
        return self.demucs.samplerate

    @property
    def demucs_num_channels(self):
        return self.demucs.audio_channels

    def _get_demucs_stems_dict(self):
        stems_dict = {}
        index = 0
        for stem in self.hparams.demucs.stems:
            stems_dict[stem] = index
            index += 1
        return stems_dict

    @staticmethod
    def load_audio(audio_path: any) -> (torch.Tensor, int):
        audio, sampling_rate = torchaudio.load(str(audio_path))
        return audio, sampling_rate

    @staticmethod
    def save_audio(audio: torch.Tensor, sampling_rate: int, audio_path: any):
        torchaudio.save(str(audio_path), audio.cpu(), sampling_rate)

    @staticmethod
    @torch.no_grad()
    def resample(waveform: torch.Tensor, orig_sampling_rate: int, new_sampling_rate: int, new_num_channels: int = 1) -> torch.Tensor:
        if waveform.dim() > 3:
            raise ValueError("Unsupported waveform shape: {}".format(waveform.shape))
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        with torch.no_grad():
            waveform = demucs.audio.convert_audio(waveform, orig_sampling_rate, new_sampling_rate, new_num_channels)
        return waveform

    @torch.inference_mode()
    def demucs_separate(self, waveform: torch.Tensor, sampling_rate: int, stems: [str]) -> torch.Tensor:
        if waveform.dim() > 3:
            raise ValueError("Unsupported waveform shape: {}".format(waveform.shape))
        if stems is None or len(stems) == 0:
            raise ValueError("No stems to be separated.")
        for s in stems:
            if s not in self.demucs_stems_dict:
                raise ValueError("Unsupported stem: {}".format(s))
        num_batches = waveform.shape[0] if waveform.dim() == 3 else 1
        num_channels = waveform.shape[-2] if waveform.dim() > 1 else 1
        if num_channels != self.demucs_num_channels or sampling_rate != self.demucs_sampling_rate:
            waveform = self.resample(waveform, sampling_rate, self.demucs_sampling_rate, self.demucs_num_channels)
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        s = demucs.apply.apply_model(self.demucs, waveform.to(self.device), shifts=0, split=True).to("cpu")
        assert s.shape == (num_batches, len(self.demucs_stems_dict), self.demucs_num_channels, waveform.shape[2])
        stems = [self.demucs_stems_dict[st] for st in stems]
        return s[:, stems, :, :]

    @staticmethod
    def normalize_loudness(audio: torch.Tensor, sampling_rate: int, target_loudness: float) -> torch.Tensor:
        assert audio.dim() in [2, 3], "Unsupported audio shape: {}.  Should be 2 or 3".format(audio.shape)
        if audio.dim() == 3:
            audio_list = [AudioUtils.normalize_loudness(audio[i], sampling_rate, target_loudness) for i in range(audio.shape[0])]
            return torch.stack(audio_list, dim=0)
        meter = pyln.Meter(sampling_rate)
        audio_npd = rearrange(audio, "c n -> n c").numpy()
        loudness = meter.integrated_loudness(audio_npd)
        audio_normalized = pyln.normalize.loudness(audio_npd, loudness, target_loudness)
        return rearrange(torch.from_numpy(audio_normalized), "n c -> c n").contiguous()

_audio_utils = None

def get_audio_utils() -> AudioUtils:
    global _audio_utils
    if _audio_utils is None:
        logger.info("Getting the singleton instance of AudioUtils.")
        _audio_utils = AudioUtils()
    return _audio_utils