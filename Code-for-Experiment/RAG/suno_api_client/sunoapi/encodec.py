from transformers import EncodecModel, AutoProcessor
import torch
import torchaudio

class EncoedecProcessor:
    def __init__(self, model_name:str="facebook/encodec_48khz", device="cuda") -> None:
        self.model = EncodecModel.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def process(self, audio_sample:torch.Tensor, sampling_rate:int):
        assert audio_sample.ndim == 2, "audio_sample must be a 2D tensor"
        assert audio_sample.shape[0] == 1 or audio_sample.shape[0] == 2, "audio_sample must have 1 or 2 channels"
        audio_device = audio_sample.device
        if sampling_rate != self.processor.sampling_rate:
            audio_sample = torchaudio.transforms.Resample(sampling_rate, self.processor.sampling_rate)(audio_sample)
        inputs = self.processor(raw_audio=audio_sample, sampling_rate=sampling_rate, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        encoder_outputs = self.model.encode(inputs["input_values"], inputs["padding_mask"])
        audio_values = self.model.decode(encoder_outputs.audio_codes, encoder_outputs.audio_scales, inputs["padding_mask"])[0]
        return audio_values.view(audio_sample.shape[0], -1).to(audio_device), self.processor.sampling_rate

    def process_file(self, audio_file:str, out_file:str):
        audio, sampling_rate = torchaudio.load(audio_file)
        processed_audio, sampling_rate = self.process(audio, sampling_rate)
        torchaudio.save(out_file, processed_audio, sampling_rate)