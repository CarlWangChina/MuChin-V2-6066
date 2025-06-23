import librosa
import laion_clap
import torch

class ClapProcessor:
    def __init__(self, device="cuda") -> None:
        self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base', tmodel='roberta', device=device)
        self.model.load_ckpt('model/music_audioset_epoch_15_esc_90.14.pt')

    @torch.no_grad()
    def processAudio(self, audio_data):
        assert len(audio_data.shape) == 1
        audio_data = audio_data.reshape(1, -1)
        audio_embed = self.model.get_audio_embedding_from_data(x=audio_data, use_tensor=False)
        return audio_embed

    @torch.no_grad()
    def processText(self, text_data: list[str]):
        text_embed = self.model.get_text_embedding(text_data, use_tensor=True)
        return text_embed

if __name__ == "__main__":
    import pyloudnorm as pyln
    p = ClapProcessor()
    text_data = ["I love the contrastive learning", "I love the pretrain model"]
    audio_data, sample_rate = librosa.load('/NAS/datasets-mp3/ali/16/1691673_src.mp3', sr=48000)
    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(audio_data)
    print(p.processAudio(audio_data).shape, loudness)
    print(p.processText(text_data).shape)