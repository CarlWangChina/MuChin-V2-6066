import torch
import numpy as np
import csv
import argparse
from pathlib import Path
from tqdm.auto import tqdm
from typing import Tuple
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger
from ama_prof_divi_vae.model import build_vae_encoder

logger = get_logger(__name__)
CHECKPOINT_FILE = Path(__file__).parent / "checkpoints" / "vae_generator_step16777_20240618.ckpt"
MEL_DATA_PATH = Path("/data/packed-mel-24000-mono")
VAE_DATA_PATH = Path("/data/packed-vae-24000-mono")

logger.info("Building the VAE encoder model...")
vae_encoder = build_vae_encoder()
if torch.cuda.is_available():
    vae_encoder = vae_encoder.cuda()
vae_encoder.eval()
vae_encoder.remove_weight_norm_()
logger.info("VAE encoder is built.")

def _load_model_checkpoint() -> Tuple[float, float]:
    ckpt = torch.load(CHECKPOINT_FILE, map_location="cpu")
    data_mean = ckpt["data_mean"]
    data_std = ckpt["data_std"]
    if "module" in ckpt:
        ckpt = ckpt["module"]
    encoder_dict = {}
    for key, value in ckpt.items():
        if key.startswith("encoder."):
            encoder_dict[key[8:]] = value
    vae_encoder.load_state_dict(encoder_dict)
    return data_mean, data_std

def main():
    arg_parser = argparse.ArgumentParser(description="Convert MEL spectrogram to VAE vectors.")
    arg_parser.add_argument("--node", "-n", type=int, default=-1, help="The node ID.")
    args = arg_parser.parse_args()
    if args.node < 0:
        logger.error("The node ID is not specified.")
        arg_parser.print_help()
        exit(1)
    logger.info("Node ID is %d.", args.node)
    try:
        logger.info(f"Loading checkpoints from {str(CHECKPOINT_FILE)}")
        data_mean, data_std = _load_model_checkpoint()
        logger.info("Checkpoint loaded.")
    except RuntimeError as e:
        logger.error(f"Failed to load checkpoint from {str(CHECKPOINT_FILE)}")
        logger.error(e)
        exit(2)
    mel_list_csv_file = MEL_DATA_PATH / f"mel_list_{args.node}.csv"
    mel_list_dat_file = MEL_DATA_PATH / f"mel_list_{args.node}.dat"
    vae_list_csv_file = VAE_DATA_PATH / f"vae_list_{args.node}.csv"
    vae_list_dat_file = VAE_DATA_PATH / f"vae_list_{args.node}.dat"
    if not mel_list_csv_file.is_file():
        logger.error(f"The csv file {mel_list_csv_file} does not exist.")
        exit(2)
    if not mel_list_dat_file.is_file():
        logger.error(f"The data file {mel_list_dat_file} does not exist.")
        exit(3)
    with open(mel_list_csv_file, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        csv_list = list(csv_reader)
        with open(mel_list_dat_file, "rb") as mel_dat_file:
            with open(vae_list_csv_file, "w", newline='') as vae_csv_file:
                vae_csv_writer = csv.writer(vae_csv_file)
                vae_csv_writer.writerow(["dataset", "song_id", "offset", "end", "length"])
                with open(vae_list_dat_file, "wb") as vae_dat_file:
                    vae_offset = 0
                    for row in tqdm(csv_list, desc="Converting"):
                        dataset = row[0]
                        song_id = row[1]
                        mel_offset = int(row[2])
                        mel_length = int(row[4])
                        assert mel_dat_file.tell() == mel_offset
                        mel_data = mel_dat_file.read(mel_length)
                        mel = torch.from_numpy(np.frombuffer(mel_data, dtype=np.float32).copy())
                        if torch.cuda.is_available():
                            mel = mel.cuda()
                        mel = mel.view(1, -1, 160).transpose(1, 2)
                        assert mel.size(1) == 160
                        try:
                            with torch.no_grad():
                                mel = (mel - data_mean) / data_std
                                vae, _ = vae_encoder(mel)
                        except RuntimeError as e:
                            logger.error(f"Failed to encode VAE: mel shape is {mel.shape}")
                            logger.error(e)
                            continue
                        vae_bytes = vae.squeeze(0).transpose(0, 1).cpu().numpy().astype(np.float32).tobytes()
                        vae_length = len(vae_bytes)
                        assert vae_length == vae.size(1) * vae.size(2) * 4
                        vae_end = vae_offset + vae_length
                        try:
                            vae_dat_file.write(vae_bytes)
                        except RuntimeError as e:
                            logger.error(f"Failed to write VAE data into packed file.")
                            logger.error(e)
                            exit(4)
                        try:
                            vae_csv_writer.writerow([dataset, song_id, vae_offset, vae_end, vae_length])
                            vae_offset = vae_end
                        except RuntimeError as e:
                            logger.error(f"Failed to update the VAE CSV file.")
                            logger.error(e)
                            exit(4)
    logger.info("All Done.")

if __name__ == "__main__":
    main()