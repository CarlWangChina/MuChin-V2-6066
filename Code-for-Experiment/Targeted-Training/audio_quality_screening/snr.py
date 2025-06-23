import torchaudio
import torch
from encodec import EncodecModel
from encodec.compress import compress, decompress
from Code_for_Experiment.Targeted_Training.audio_quality_screening.encodec.utils import convert_audio

def encodec_model_48khz(checkpoint_name="model/encodec_48khz-7e698e3e.th"):
    target_bandwidths = [3.0, 6.0, 12.0, 24.0]
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
        name="encodec_48khz",
    )
    state_dict = torch.load(checkpoint_name, map_location=torch.device("mps"))
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = encodec_model_48khz()

def snr(raw_audio):
    wav, raw_sr = torchaudio.load(raw_audio)
    wav_inp = convert_audio(wav, raw_sr, model.sample_rate, model.channels)
    f = compress(model=model, wav=wav_inp)
    rec_wav, _ = decompress(model=model, compressed=f)
    signal_energy = torch.sum(wav_inp**2, dim=(0, 1))
    noise_energy = torch.sum(rec_wav**2, dim=(0, 1))
    r = 10 * torch.log10(signal_energy / noise_energy)
    return r.item()

if __name__ == "__main__":
    save_path = "6548e6b616551961066946dd97269807535a1eae_src.mp3"
    print(snr(save_path))