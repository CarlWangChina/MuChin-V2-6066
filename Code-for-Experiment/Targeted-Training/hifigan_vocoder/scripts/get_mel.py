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
from Code_for_Experiment.Targeted_Training.hifigan_vocoder.ama-prof-divi_hifigan.mel.builder import build_mel_generator
mel_generator = build_mel_generator()
if torch.cuda.is_available():
    mel_generator = mel_generator.cuda()
logger.info("Mel generator is built.")

def main():
    arg_parser = argparse.ArgumentParser(description="Get mel spectrogram from audio file.")
    arg_parser.add_argument("--input", "-i", type=str, help="The input audio file.")
    arg_parser.add_argument("--output", "-o", type=str, help="The output mel file.")
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
    try:
        audio, sr = torchaudio.load(args.input)
        if torch.cuda.is_available():
            audio = audio.cuda()
    except RuntimeError as e:
        logger.error(f"Failed to load the audio file: {args.input}")
        logger.error(e)
        exit(3)
    logger.info(f"Loaded the audio file {args.input}.")
    logger.info(f"Sample rate: {sr}, shape: {audio.shape}")
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if sr != mel_generator.sampling_rate:
        try:
            resampler = torchaudio.transforms.Resample(sr, mel_generator.sampling_rate)
            if torch.cuda.is_available():
                resampler = resampler.cuda()
            audio = resampler(audio)
            logger.info(f"Resampled the audio file to {mel_generator.sampling_rate} sps")
        except RuntimeError as e:
            logger.error(f"Failed to resample the audio file: {args.input}")
            logger.error(e)
            exit(3)
    audio = audio.mean(dim=0, keepdim=True)
    try:
        mel = mel_generator(audio).squeeze(0)
    except RuntimeError as e:
        logger.error(f"Failed to generate mel spectrogram.")
        logger.error(e)
        exit(4)
    logger.info(f"MEL spectrogram generated. {audio.shape} --> Mel spectrogram shape: {mel.shape}")
    try:
        torch.save({"mel": mel}, args.output)
        logger.info(f"MEL spectrogram saved as {args.output}.")
    except RuntimeError as e:
        logger.error(f"Failed to save mel spectrogram to file.")
        logger.error(e)
        exit(5)

if __name__ == "__main__":
    main()