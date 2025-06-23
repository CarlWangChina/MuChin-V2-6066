import sys
from pathlib import Path
import argparse
import torch
import torch.nn.functional as F
import torchaudio
import laion_clap
from ama_prof_divi_common.utils import get_logger
from ama_prof_divi_hifigan.model import build_generator
from ama_prof_divi_hifigan.mel import build_mel_generator
from ama_prof_divi_vae.model import build_vae_decoder, build_vae_encoder
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.models.music_dit import MusicDiTModel
from ama_prof_divi_text.phoneme import Lyrics2Phoneme, PhonemeTokenizer

logger = get_logger(__name__)
current_path = Path(__file__).absolute()
root_path = current_path.parent.parent
sys.path.append(str(root_path))

HIFIGAN_MODEL_CKPT = Path(__file__).parent.parent / "checkpoints/hifigan-v1-large-20240603.ckpt"
VAE_MODEL_CKPT = Path(__file__).parent.parent / "checkpoints/vae_generator_step16777_20240618.ckpt"
CLAP_MODEL_CKPT = Path(__file__).parent.parent / "checkpoints/laion_clap_music_epoch_15_esc_90.14.ckpt"
DIT_CHECKPOINT = Path(__file__).parent.parent / "checkpoints/music_dit_step_82280.ckpt"
HIFIGAN_VERSION = "v1_large"
CLAP_CHUNK_LENGTH = 65536 * 2
VAE_TO_CLAP_RATIO = 16

def _align_clap_to_vae(clap: torch.Tensor, seq_len: int) -> torch.Tensor:
    batch_size = clap.size(0)
    num_frames = clap.size(1)
    clap_out = torch.zeros(batch_size, seq_len, clap.size(-1), device=clap.device).type_as(clap)
    for i in range(seq_len):
        idx = min(i // VAE_TO_CLAP_RATIO, num_frames - 1)
        clap_out[:, i, :] = clap[:, idx, :]
    return clap_out

def main():
    arg_parser = argparse.ArgumentParser(description="Generate VAE sequence by DIT model.")
    arg_parser.add_argument("--duration", "-d", type=float, default=120.0, help="Duration of the generated audio in seconds.")
    arg_parser.add_argument("--reference", "-r", type=str, help="Reference audio file for the generation.")
    arg_parser.add_argument("--lyrics", "-l", type=str, help="Lyrics for the generation.")
    arg_parser.add_argument("--output", "-o", type=str, help="Output audio file.")
    arg_parser.add_argument("--model", "-m", type=str, help="Pretrained DIT model checkpoint.", default=str(DIT_CHECKPOINT))
    arg_parser.add_argument("--cpu", "-c", type=int, help="Use CPU for generation.", default=0)
    arg_parser.add_argument("--shallow", "-s", type=int, help="Use shallow diffusion steps.")
    args = arg_parser.parse_args()

    if args.duration <= 0:
        logger.error("Duration should be positive.")
        exit(1)
    if args.output is None:
        logger.error("The output audio file is not specified.")
        arg_parser.print_help()
        exit(1)
    if args.reference is None:
        logger.warn("Reference audio file is not specified. The generation will be based on random noise.")
    elif not Path(args.reference).exists():
        logger.error("Reference audio file does not exist.")
        exit(1)
    if args.lyrics is None:
        logger.warn("Lyrics is not specified. The generation will be lyricless.")
    elif not Path(args.lyrics).exists():
        logger.error(f"Lyrics file {args.lyrics} does not exist.")
        exit(1)

    logger.info("Building the HifiGAN generator...")
    hifigan_generator = build_generator(HIFIGAN_VERSION)
    if (not args.cpu) and torch.cuda.is_available():
        hifigan_generator = hifigan_generator.cuda()
    try:
        logger.info(f"Loading the HifiGAN model from {HIFIGAN_MODEL_CKPT}.")
        ckpt = torch.load(HIFIGAN_MODEL_CKPT, map_location="cpu")
        if "module" in ckpt:
            hifigan_generator.load_state_dict(ckpt["module"])
        else:
            hifigan_generator.load_state_dict(ckpt)
    except RuntimeError as e:
        logger.error(f"Failed to load the HifiGAN model file {HIFIGAN_MODEL_CKPT}.")
        logger.error(e)
        exit(3)
    hifigan_generator.eval()
    hifigan_generator.remove_weight_norm_()

    logger.info("Building the VAE decoder model...")
    vae_decoder = build_vae_decoder()
    if (not args.cpu) and torch.cuda.is_available():
        vae_decoder = vae_decoder.cuda()
    try:
        logger.info(f"Loading the VAE model checkpoint {VAE_MODEL_CKPT}...")
        ckpt = torch.load(VAE_MODEL_CKPT, map_location="cpu")
        data_mean = ckpt["data_mean"] if 'data_mean' in ckpt else DEFAULT_MEL_MEAN
        data_std = ckpt["data_std"] if 'data_std' in ckpt else DEFAULT_MEL_STD
        if "module" in ckpt:
            ckpt = ckpt["module"]
        decoder_dict = {}
        for key, value in ckpt.items():
            if key.startswith("decoder."):
                decoder_dict[key[8:]] = value
        vae_decoder.load_state_dict(decoder_dict)
    except RuntimeError as e:
        logger.error(f"Failed to load the model file {VAE_MODEL_CKPT}.")
        logger.error(e)
        exit(3)
    vae_decoder.eval()
    vae_decoder.remove_weight_norm_()

    logger.info("VAE encoder model is built.")
    logger.info("Building the DIT model...")
    dit_model = MusicDiTModel()
    try:
        logger.info(f"Loading the DIT mode checkpoint {args.model}...")
        ckpt = torch.load(args.model, map_location="cpu")
        if "module" in ckpt:
            ckpt = ckpt["module"]
        dit_model.load_state_dict(ckpt)
    except RuntimeError as e:
        logger.error(f"Failed to load the model file {VAE_MODEL_CKPT}.")
        logger.error(e)
        exit(3)
    if (not args.cpu) and torch.cuda.is_available():
        dit_model = dit_model.cuda()
    dit_model.eval()
    logger.info("DIT model is built.")

    logger.info("Building the CLAP model...")
    logger.info("Building the Clap model")
    clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", tmodel="roberta")
    try:
        logger.info(f"Loading the CLAP model checkpoint {CLAP_MODEL_CKPT}...")
        clap_model.load_ckpt(str(CLAP_MODEL_CKPT))
    except RuntimeError as e:
        logger.error(f"Failed to load the model file {CLAP_MODEL_CKPT}.")
        logger.error(e)
        exit(3)
    clap_model.eval()
    if (not args.cpu) and torch.cuda.is_available():
        clap_model = clap_model.cuda()

    seq_len = int(args.duration * 16)
    if args.reference is not None:
        try:
            ref_audio, ref_sampling_rate = torchaudio.load(args.reference)
        except RuntimeError as e:
            logger.error(f"Failed to load the reference audio file {args.reference}.")
            logger.error(e)
            exit(3)
        clap_sampling_rate = clap_model.model_cfg["audio_cfg"]["sample_rate"]
        clap_dim = clap_model.model_cfg["text_cfg"]["width"]
        if ref_sampling_rate != clap_sampling_rate:
            resampler = torchaudio.transforms.Resample(ref_sampling_rate, clap_sampling_rate)
            if (not args.cpu) and torch.cuda.is_available():
                resampler = resampler.cuda()
            ref_audio = resampler(ref_audio)
            if ref_audio.size(0) > 1:
                ref_audio = ref_audio.mean(dim=0, keepdim=True)
        logger.info("Getting the CLAP embedding for the reference audio...")
        len = ref_audio.size(-1)
        num_chunks = (len + CLAP_CHUNK_LENGTH - 1) // CLAP_CHUNK_LENGTH
        ref_clap = torch.zeros(1, num_chunks, clap_dim)
        if (not args.cpu) and torch.cuda.is_available():
            ref_clap = ref_clap.cuda()
        for i in range(num_chunks):
            chunk = ref_audio[:, i * CLAP_CHUNK_LENGTH: (i + 1) * CLAP_CHUNK_LENGTH]
            if chunk.size(-1) < CLAP_CHUNK_LENGTH:
                chunk = F.pad(chunk, (0, CLAP_CHUNK_LENGTH - chunk.size(-1)))
            with torch.no_grad():
                clap = clap_model.get_audio_embedding_from_data(chunk, use_tensor=True)
            assert clap.size() == (1, clap_dim)
            ref_clap[:, i, :] = clap
        ref_clap = _align_clap_to_vae(ref_clap, seq_len)
        logger.info("ref_clap: %s", ref_clap.size())

        if args.shallow is not None:
            assert 0 <= args.shallow < dit_model.num_inference_timesteps, "Invalid shallow diffusion steps."
            logger.info(f"Use shallow diffusion for {dit_model.num_inference_timesteps - args.shallow - 1} steps.")
            try:
                ref_audio, ref_sampling_rate = torchaudio.load(args.reference)
            except RuntimeError as e:
                logger.error(f"Failed to load the reference audio file {args.reference}.")
                logger.error(e)
                exit(3)
            mel_generator = build_mel_generator()
            if (not args.cpu) and torch.cuda.is_available():
                mel_generator = mel_generator.cuda()
                ref_audio = ref_audio.cuda()
            vae_encoder = build_vae_encoder()
            if (not args.cpu) and torch.cuda.is_available():
                vae_encoder = vae_encoder.cuda()
            try:
                logger.info(f"Loading the VAE model checkpoint {VAE_MODEL_CKPT}...")
                ckpt = torch.load(VAE_MODEL_CKPT, map_location="cpu")
                if "module" in ckpt:
                    ckpt = ckpt["module"]
                encoder_dict = {}
                for key, value in ckpt.items():
                    if key.startswith("encoder."):
                        encoder_dict[key[8:]] = value
                vae_encoder.load_state_dict(encoder_dict)
            except RuntimeError as e:
                logger.error(f"Failed to load the model file {args.model}.")
                logger.error(e)
                exit(3)
            if ref_sampling_rate != mel_generator.sampling_rate:
                try:
                    resampler = torchaudio.transforms.Resample(ref_sampling_rate, mel_generator.sampling_rate)
                    if (not args.cpu) and torch.cuda.is_available():
                        resampler = resampler.cuda()
                    ref_audio = resampler(ref_audio)
                except RuntimeError as e:
                    logger.error(f"Failed to resample the reference audio")
                    logger.error(e)
                    exit(3)
            ref_audio = ref_audio.mean(dim=0, keepdim=True)
            try:
                ref_mel = mel_generator(ref_audio)
            except RuntimeError as e:
                logger.error(f"Failed to generate mel spectrogram for refernece audio.")
                logger.error(e)
                exit(4)
            logger.info(f"MEL spectrogram for reference model: {ref_mel.shape}")
            ref_mel = (ref_mel - data_mean) / data_std
            ref_vae_latent, _ = vae_encoder(ref_mel)
            ref_vae_latent = ref_vae_latent.transpose(1, 2)
            if ref_vae_latent.size(1) > seq_len:
                ref_vae_latent = ref_vae_latent[:, :seq_len, :]
            elif ref_vae_latent.size(1) < seq_len:
                ref_vae_latent = F.pad(ref_vae_latent, (0, 0, 0, seq_len - ref_vae_latent.size(1)))
            logger.info(f"VAE latent space for reference model: {ref_vae_latent.shape}")
        else:
            ref_vae_latent = None
    else:
        ref_clap = None
        ref_vae_latent = None

    if args.lyrics is not None:
        lyrics_to_phoneme = Lyrics2Phoneme()
        phome_tokenizer = PhonemeTokenizer()
        try:
            with open(args.lyrics, "r") as f:
                lyrics = f.read()
                phoneme = lyrics_to_phoneme.translate(lyrics)
                phoneme_tokens = phome_tokenizer.tokenize(phoneme).unsqueeze(0)
                if (not args.cpu) and torch.cuda.is_available():
                    phoneme_tokens = phoneme_tokens.cuda()
                logger.info("phoneme tokens: %s", phoneme_tokens.size())
        except RuntimeError as e:
            logger.error(f"Failed to read the lyrics file {args.lyrics}.")
            logger.error(e)
            exit(3)
    else:
        phoneme_tokens = None

    logger.info("Generating audio...")
    with torch.no_grad():
        dit_model.eval()
        if args.shallow is None:
            x = torch.randn(1, seq_len, 512)
            if (not args.cpu) and torch.cuda.is_available():
                x = x.cuda()
            vae = dit_model.inference(x=x, clap=ref_clap, lyrics=phoneme_tokens, cfg_scale=0.5).transpose(1, 2)
        else:
            logger.info("Using shallow diffusion for %d steps.", args.shallow)
            vae = dit_model.shallow_diffusion(vae_ref=ref_vae_latent, clap=ref_clap, lyrics=phoneme_tokens, start_timestep_index=dit_model.num_inference_timesteps - args.shallow - 1, cfg_scale=0.5).transpose(1, 2)
        logger.info("Generated VAE: %s", vae.size())
        torch.save(vae, "data/vae.pt")
        mel = vae_decoder(vae)
        mel = mel * data_std + data_mean
        logger.info("Decoded mel: %s", mel.size())
        audio = hifigan_generator(mel).squeeze(0)
        logger.info("Generated audio: %s", audio.size())
    try:
        torchaudio.save(args.output, audio.cpu(), sample_rate=24000)
        logger.info(f"Audio waveform saved to {args.output}.")
    except RuntimeError as e:
        logger.error(f"Failed to save output audio to file.")
        logger.error(e)
        exit(5)
    logger.info("Done.")

if __name__ == "__main__":
    main()