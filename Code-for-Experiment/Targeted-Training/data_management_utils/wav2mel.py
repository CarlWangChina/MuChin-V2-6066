import torch
import numpy as np
import csv
import argparse
from pathlib import Path
from tqdm.auto import tqdm
from ama_prof_divi_common.utils import get_logger
from Code_for_Experiment.Targeted_Training.hifigan_vocoder.ama-prof-divi_hifigan.mel.builder import build_mel_generator

logger = get_logger(__name__)
WAV_DATA_PATH = Path("/data/packed-wav-24000-mono")
MEL_DATA_PATH = Path("/data/packed-mel-24000-mono")
mel_generator = build_mel_generator()
if torch.cuda.is_available():
    mel_generator = mel_generator.cuda()
logger.info("Mel generator is built.")

def main():
    arg_parser = argparse.ArgumentParser(description="Convert audio to mel spectrogram.")
    arg_parser.add_argument("--node", "-n", type=int, default=-1, help="The node ID.")
    args = arg_parser.parse_args()
    if args.node < 0:
        logger.error("The node ID is not specified.")
        arg_parser.print_help()
        exit(1)
    logger.info("Node ID is %d.", args.node)
    wav_list_csv_file = WAV_DATA_PATH / f"wav_list_{args.node}.csv"
    mel_list_csv_file = MEL_DATA_PATH / f"mel_list_{args.node}.csv"
    wav_list_dat_file = WAV_DATA_PATH / f"wav_list_{args.node}.dat"
    mel_list_dat_file = MEL_DATA_PATH / f"mel_list_{args.node}.dat"
    if not wav_list_csv_file.is_file():
        logger.error(f"The csv file {wav_list_csv_file} does not exist.")
        exit(2)
    if not wav_list_dat_file.is_file():
        logger.error(f"The data file {wav_list_dat_file} does not exist.")
        exit(3)
    with open(wav_list_csv_file, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        csv_list = list(csv_reader)
        with open(wav_list_dat_file, "rb") as wav_dat_file:
            with open(mel_list_csv_file, "w") as mel_csv_file:
                mel_csv_writer = csv.writer(mel_csv_file)
                mel_csv_writer.writerow(["dataset", "song_id", "offset", "end", "length"])
                with open(mel_list_dat_file, "wb") as mel_dat_file:
                    mel_offset = 0
                    for row in tqdm(csv_list, desc="Converting"):
                        dataset = row[0]
                        song_id = row[1]
                        wav_offset = int(row[2])
                        wav_length = int(row[4])
                        assert wav_dat_file.tell() == wav_offset
                        wav_data = wav_dat_file.read(wav_length)
                        audio = torch.from_numpy(np.frombuffer(wav_data, dtype=np.int16).copy()).float()
                        audio = (audio / 32768.0).unsqueeze(0).unsqueeze(0)
                        if torch.cuda.is_available():
                            audio = audio.cuda()
                        mel = mel_generator(audio).squeeze(0).squeeze(0).transpose(0, 1)
                        assert mel.size(1) == 160
                        mel_bytes = mel.flatten().cpu().numpy().tobytes()
                        mel_length = len(mel_bytes)
                        assert mel_length == mel.size(0) * mel.size(1) * 4
                        mel_end = mel_offset + mel_length
                        mel_dat_file.write(mel_bytes)
                        mel_csv_writer.writerow([dataset, song_id, mel_offset, mel_end, mel_length])
                        mel_offset = mel_end

if __name__ == "__main__":
    main()