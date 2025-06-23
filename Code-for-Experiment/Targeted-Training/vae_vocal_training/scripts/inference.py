import sys
sys.path.append("/home/male/codes/ama-prof-divi_vae_singing")
import argparse
import torch
import torchaudio
import pyloudnorm as pyln
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger
from ama_prof_divi_vae2.model import VAE2Model
from pathlib import Path

logger = get_logger(__name__)
NORMALIZED_LOUDNESS = -14.0
current_path = Path(__file__).absolute()
root_path = current_path.parent.parent
sys.path.append(str(root_path))
DEFAULT_MODEL_CKPT = '/home/male/codes/ama-prof-divi_vae_singing/g/global_step2562/mp_rank_00_model_states.pt'

def main():
    arg_parser = argparse.ArgumentParser(description="Encode and decode by the VAE model.")
    arg_parser.add_argument("--inputs", "-i", nargs="+", type=str, help="The input audio files.")
    arg_parser.add_argument("--output", "-o", type=str, help="The output audio file.")
    arg_parser.add_argument("--model", "-m", type=str, help="The model file.", default=str(DEFAULT_MODEL_CKPT))
    arg_parser.add_argument("--cpu", "-c", type=int, help="Use CPU only.", default=0)
    arg_parser.add_argument("--noise", "-n", type=int, help="Noise level.", default=0)
    arg_parser.add_argument("--reparam", "-r", type=int, help="Use reparameterize.", default=1)
    args = arg_parser.parse_args()

    if args.inputs is None:
        logger.error("No input audio file is specified.")
        arg_parser.print_help()
        exit(1)

    for input_file in args.inputs:
        if not Path(input_file).is_file():
            logger.error(f"Input audio file does not exist: {input_file}")
            exit(1)

    if args.output is None:
        logger.error("The output audio file is not specified.")
        arg_parser.print_help()
        exit(1)

    if not Path(args.output).parent.is_dir():
        logger.error(f"The output directory does not exist: {str(Path(args.output).parent)}")
        exit(2)

    if args.noise < 0 or args.noise >= 1000:
        logger.error("The noise level should be between 0 and 999.")
        exit(3)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    if args.noise > 0:
        betas = torch.linspace(1e-4, 2e-2, 1000, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_betas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    else:
        sqrt_alphas_cumprod = None
        sqrt_betas_cumprod = None

    logger.info("Building the model...")
    model = VAE2Model().to(device)

    try:
        logger.info(f"Loading the VAE model checkpoint {args.model}...")
        ckpt = torch.load(args.model, map_location="cpu")
        if "module" in ckpt:
            ckpt = ckpt["module"]
        model.load_state_dict(ckpt)
    except RuntimeError as e:
        logger.error(f"Failed to load the model file {args.model}.")
        logger.error(e)
        exit(3)

    model.eval()
    model.remove_weight_norm_()
    logger.info("VAE model is built.")
    meter = pyln.Meter(model.sampling_rate)
    logger.info("Reading input audio files...")
    input_audios = []
    min_length = 1e10
    for input_file in args.inputs:
        waveform, sample_rate = torchaudio.load(input_file)
        if sample_rate != model.sampling_rate:
            try:
                resampler = torchaudio.transforms.Resample(sample_rate, model.sampling_rate)
                waveform = resampler(waveform)
                if waveform.size(0) > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                waveform = waveform.transpose(0, 1).numpy()
                loudness = meter.integrated_loudness(waveform)
                waveform = pyln.normalize.loudness(waveform, loudness, NORMALIZED_LOUDNESS)
                waveform = torch.from_numpy(waveform).transpose(0, 1)
                print(waveform.shape)
            except RuntimeError as e:
                logger.error(f"Failed to resample the audio file: {input_file}")
                logger.error(e)
                exit(3)
            waveform = waveform.mean(dim=0, keepdim=True)
            input_audios.append(waveform)
            if waveform.size(1) < min_length:
                min_length = waveform.size(1)

    for i in range(len(input_audios)):
        input_audios[i] = input_audios[i][:, :min_length]

    input_audios = torch.cat(input_audios, dim=0).to(device)
    input_audio = input_audios.mean(dim=0, keepdim=True).unsqueeze(0)
    del input_audios

    logger.info("Encoding the input audio...")
    try:
        with torch.no_grad():
            mean, logvar = model.encode(input_audio)
        logger.info("Encoded: %s -> %s", input_audio.size(), mean.size())
    except RuntimeError as e:
        logger.error("Failed to encode the input audio.")
        logger.error(e)
        exit(3)

    if args.reparam:
        logger.info("Reparameterizing the latent variables...")
        with torch.no_grad():
            std = torch.exp(0.5 * logvar)
            logger.info("mean: %s, std: %s", mean.mean().item(), std.mean().item())
            eps = torch.randn_like(std)
            z = mean + eps * std
    else:
        z = mean

    if args.noise > 0:
        logger.info("Adding noise to the latent variables...")
        with torch.no_grad():
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[args.noise].to(device)
            sqrt_betas_cumprod_t = sqrt_betas_cumprod[args.noise].to(device)
            eps = torch.randn_like(z)
            z = sqrt_alphas_cumprod_t * z + sqrt_betas_cumprod_t * eps

    logger.info("Decoding the latent variables...")
    try:
        with torch.no_grad():
            output_audio = model.decode(z)
        logger.info("Decoded: %s -> %s", z.size(), output_audio.size())
    except RuntimeError as e:
        logger.error("Failed to decode the latent variables.")
        logger.error(e)
        exit(3)

    logger.info("Writing the output audio file...")
    try:
        output_audio = output_audio.squeeze(0).transpose(0, 1).cpu().numpy()
        loudness = meter.integrated_loudness(output_audio)
        output_audio = pyln.normalize.loudness(output_audio, loudness, NORMALIZED_LOUDNESS)
        output_audio = torch.from_numpy(output_audio).transpose(0, 1)
        torchaudio.save(args.output, output_audio.cpu(), model.sampling_rate)
    except RuntimeError as e:
        logger.error("Failed to write the output audio file.")
        logger.error(e)
        exit(3)

    logger.info("All done.")

if __name__ == "__main__":
    main()