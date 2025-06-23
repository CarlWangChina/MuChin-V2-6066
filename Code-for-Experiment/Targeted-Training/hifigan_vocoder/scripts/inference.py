import sys
from pathlib import Path
import argparse
import torch
import torchaudio
from ama_prof_divi_common.utils import get_logger
logger = get_logger(__name__)
current_path = Path(__file__).absolute()
root_path = current_path.parent.parent
sys.path.append(str(root_path))
from Code_for_Experiment.Targeted_Training.hifigan_vocoder.ama-prof-divi_hifigan.model.builder import build_generator
DEFAULT_VERSION = "v1_large"
DEFAULT_MODEL_CKPT = Path(__file__).parent.parent / "checkpoints/hifigan-v1-large-20240603.ckpt"

def main():
    arg_parser = argparse.ArgumentParser(description="Get mel spectrogram from audio file.")
    arg_parser.add_argument("--input", "-i", type=str, help="The input audio file.")
    arg_parser.add_argument("--output", "-o", type=str, help="The output mel file.")
    arg_parser.add_argument("--model", "-m", type=str, help="The model file.", default=str(DEFAULT_MODEL_CKPT))
    arg_parser.add_argument("--version", "-v", type=str, default=DEFAULT_VERSION)
    args = arg_parser.parse_args()
    if args.input is None:
        logger.error("The input audio file is not specified.")
        arg_parser.print_help()
        exit(1)
    if args.output is None:
        logger.error("The output audio file is not specified.")
        arg_parser.print_help()
        exit(1)
    if not Path(args.input).is_file():
        logger.error(f"The input audio file does not exist: {args.input}")
        exit(2)
    if not Path(args.output).parent.is_dir():
        logger.error(f"The output directory does not exist: {str(Path(args.output).parent)}")
        exit(2)
    logger.info("Building the HifiGAN generator model...")
    generator = build_generator(version=args.version)
    if torch.cuda.is_available():
        generator = generator.cuda()
    try:
        ckpt = torch.load(args.model, map_location="cpu")
        if "module" in ckpt:
            generator.load_state_dict(ckpt["module"])
        else:
            generator.load_state_dict(ckpt)
    except RuntimeError as e:
        logger.error(f"Failed to load the model file {args.model}.")
        logger.error(e)
        exit(3)
    generator.eval()
    generator.remove_weight_norm_()
    logger.info("Generator model is built.")
    try:
        mel = torch.load(args.input, map_location="cpu")["mel"]
        if torch.cuda.is_available():
            mel = mel.cuda()
        assert mel.dim() == 2 or mel.dim() == 3, "The mel spectrogram should be 2D or 3D tensor. Actual shape: {mel.shape}"
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        assert mel.size(-2) == generator.n_mels, f"The number of mel bands in the input mel spectrogram is {mel.size(-2)}, which is different from the model's n_mels {generator.n_mels}."
        logger.info(f"Mel spectrogram loaded.  Shape: {mel.shape}")
    except RuntimeError as e:
        logger.error(f"Failed to load the mel spectrogram file {args.input}.")
        logger.error(e)
        exit(3)
    try:
        with torch.no_grad():
            audio = generator(mel).squeeze(0)
        audio = audio.clamp(-1.0, 1.0)
        logger.info(f"Audio waveform generated.  Shape: {audio.shape}")
    except RuntimeError as e:
        logger.error("Failed to generate audio waveform.")
        logger.error(e)
        exit(4)
    try:
        torchaudio.save(args.output, audio.cpu(), sample_rate=24000)
        logger.info(f"Audio waveform saved to {args.output}.")
    except RuntimeError as e:
        logger.error(f"Failed to save output audio to file.")
        logger.error(e)
        exit(5)

if __name__ == "__main__":
    main()