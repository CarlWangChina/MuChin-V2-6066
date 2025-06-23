import torch
import torch.nn as nn
from typing import Optional
from einops import rearrange
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.logging import get_logger
import get_phoneme_tokenizer
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.modules.fastspeech2.fs2 import FastSpeech2
from ama_prof_divi.modules.diffusion import LatentDiffusion, DiffusionArgs, WaveNetModelArgs
from .vocoders.vocoder import SVSVocoder

logger = get_logger(__name__)

class SVSModel(nn.Module):
    def __init__(self, hparams: dict):
        super(SVSModel, self).__init__()
        self.device = hparams["ama-prof-divi"]["device"]
        self.fs2_hparams = hparams["ama-prof-divi"]["models"]["vocal"]["fs2"]
        self.diff_hparams = hparams["ama-prof-divi"]["models"]["vocal"]["diffusion"]
        self.vocoder = SVSVocoder(hparams)
        self.phoneme_tokenizer = get_phoneme_tokenizer(hparams)
        self.fs2 = FastSpeech2(
            phoneme_vocab_size=self.phoneme_tokenizer.vocab_size,
            hidden_size=self.fs2_hparams["hidden_size"],
            num_encoder_layers=self.fs2_hparams["num_encoder_layers"],
            num_decoder_layers=self.fs2_hparams["num_decoder_layers"],
            kernel_size=self.fs2_hparams["kernel_size"],
            num_heads=self.fs2_hparams["num_heads"],
            output_dim=self.vocoder.num_mel_bands,
            duration_predictor_layers=self.fs2_hparams["duration_predictor"]["layers"],
            duration_predictor_kernel_size=self.fs2_hparams["duration_predictor"]["kernel_size"],
            pitch_predictor_layers=self.fs2_hparams["pitch_predictor"]["layers"],
            pitch_predictor_kernel_size=self.fs2_hparams["pitch_predictor"]["kernel_size"],
            energy_predictor_layers=self.fs2_hparams["energy_predictor"]["layers"],
            energy_predictor_kernel_size=self.fs2_hparams["energy_predictor"]["kernel_size"],
            phoneme_padding_token_id=self.phoneme_tokenizer.pad_id,
            dropout=self.fs2_hparams["dropout"],
            num_speakers=self.fs2_hparams["num_speakers"],
            melody_feature_rate=hparams["ama-prof-divi"]["models"]["semantic"]["encoder"]["features_rate"],
            sampling_rate=self.vocoder.sampling_rate,
            vocoder_hop_size=self.vocoder.hop_size,
            norm_type=self.fs2_hparams["norm"],
            padding_type=self.fs2_hparams["padding"],
            act_type=self.fs2_hparams["activation"],
            device=self.device
        )
        self.diffusion_enabled = self.diff_hparams["enabled"]
        if self.diffusion_enabled:
            self.register_buffer("spec_norm_enabled", torch.BoolTensor([False]))
            self.register_buffer("spec_min", torch.zeros(self.vocoder.num_mel_bands,))
            self.register_buffer("spec_max", torch.zeros(self.vocoder.num_mel_bands,))
            assert self.diff_hparams["denoiser"] == "wavenet", \
                f"The SVS model currently only supports wavenet as the denoiser. "
            wavenet_args = WaveNetModelArgs(
                in_channels=self.vocoder.num_mel_bands,
                out_channels=self.vocoder.num_mel_bands,
                model_channels=self.diff_hparams["wavenet"]["model_channels"],
                context_channels=self.fs2_hparams["hidden_size"],
                num_layers=self.diff_hparams["wavenet"]["num_layers"],
                dilation_cycle=self.diff_hparams["wavenet"]["dilation_cycle"],
            )
            diffusion_args = DiffusionArgs(
                sampler=self.diff_hparams["sampler"]["name"],
                sampler_extra_args=self.diff_hparams["sampler"],
                denoiser="wavenet",
                wavenet=wavenet_args
            )
            self.diffusion = LatentDiffusion(diffusion_args, device=self.device)

    @torch.inference_mode()
    def generate(self, phoneme_tokens: torch.Tensor, pitch_tokens: torch.Tensor, phoneme_duration: Optional[torch.Tensor] = None, speaker_ids: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        mel, context = self.fs2(phoneme_tokens=phoneme_tokens, phoneme_duration=phoneme_duration, pitch_tokens=pitch_tokens, speaker_ids=speaker_ids, mask=mask)
        mel = mel.detach()
        context = context.detach()
        if self.diffusion_enabled:
            if self.spec_norm_enabled.item():
                mel = (mel - self.spec_min) / (self.spec_max - self.spec_min) * 2.0 - 1.0
                mel = rearrange(mel, "b t c -> b c t")
                mel = self.diffusion.generate(seq_len=mel.shape[2], context=context, latent_start=mel, description="Generating SVS mel-spectrogram")
                mel = rearrange(mel, "b c t -> b t c")
        return mel