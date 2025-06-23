import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import csv
import argparse
import laion_clap
from pathlib import Path
from tqdm.auto import tqdm
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger

logger = get_logger(__name__)
CHECKPOINT_FILE = Path(__file__).parent / "checkpoints" / "laion_clap_music_epoch_15_esc_90.14.ckpt"
WAV_DATA_PATH = Path("/data/packed-wav-24000-mono")
CLAP_DATA_PATH = Path("/data/packed-clap-24000-mono")
DEVICE = torch.device("cuda:1")
CHUNK_LENGTH = 65536 * 2

logger.info("Building the Clap model")
clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", tmodel="roberta").to(DEVICE)
clap_model.load_ckpt(str(CHECKPOINT_FILE))
clap_model.eval()
logger.info(f"Loaded pre-trained model from {str(CHECKPOINT_FILE)}")

def get_clap_sampling_rate() -> int:
    return clap_model.model_cfg["audio_cfg"]["sample_rate"]

def get_clap_dim() -> int:
    return clap_model.model_cfg["text_cfg"]["width"]

def get_clap_vector(audio: torch.Tensor) -> torch.Tensor:
    length = audio.size(-1)
    num_chunks = (length + CHUNK_LENGTH - 1) // CHUNK_LENGTH
    result = torch.zeros(1, num_chunks, get_clap_dim(), device=DEVICE)
    for i in range(num_chunks):
        chunk = audio[:, i * CHUNK_LENGTH: (i + 1) * CHUNK_LENGTH]
        if chunk.size(-1) < CHUNK_LENGTH:
            chunk = F.pad(chunk, (0, CHUNK_LENGTH - chunk.size(-1)))
        with torch.no_grad():
            clap = clap_model.get_audio_embedding_from_data(chunk, use_tensor=True)
        assert clap.size() == (1, get_clap_dim())
        result[:, i, :] = clap
    return result

def main():
    arg_parser = argparse.ArgumentParser(description="Get Clap vector from waveforms.")
    arg_parser.add_argument("--node", "-n", type=int, default=-1, help="The node ID.")
    args = arg_parser.parse_args()
    if args.node < 0:
        logger.error("The node ID is not specified.")
        arg_parser.print_help()
        exit(1)
    logger.info("Node ID is %d.", args.node)
    logger.info("CLAP sampling rate: %d", get_clap_sampling_rate())
    logger.info("CLAP dimension: %d", get_clap_dim())
    resampler = torchaudio.transforms.Resample(24000, get_clap_sampling_rate()).to(DEVICE)
    wav_list_csv_file = WAV_DATA_PATH / f"wav_list_{args.node}.csv"
    wav_list_dat_file = WAV_DATA_PATH / f"wav_list_{args.node}.dat"
    clap_list_csv_file = CLAP_DATA_PATH / f"clap_list_{args.node}.csv"
    clap_list_dat_file = CLAP_DATA_PATH / f"clap_list_{args.node}.dat"
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
            with open(clap_list_csv_file, "w") as clap_csv_file:
                clap_csv_writer = csv.writer(clap_csv_file)
                clap_csv_writer.writerow(["dataset", "song_id", "offset", "end", "length"])
                with open(clap_list_dat_file, "wb") as clap_dat_file:
                    clap_offset = 0
                    for row in tqdm(csv_list, desc="Converting"):
                        dataset = row[0]
                        song_id = row[1]
                        wav_offset = int(row[2])
                        wav_length = int(row[4])
                        assert wav_dat_file.tell() == wav_offset
                        wav_data = wav_dat_file.read(wav_length)
                        audio = torch.from_numpy(np.frombuffer(wav_data, dtype=np.int16).copy()).float()
                        audio = (audio / 32768.0).unsqueeze(0).to(DEVICE)
                        try:
                            with torch.no_grad():
                                audio = resampler(audio)
                        except Exception as e:
                            logger.error("Error in resampling: %s", song_id)
                            logger.error(e)
                            continue
                        try:
                            clap = get_clap_vector(audio).squeeze(0)
                        except Exception as e:
                            logger.error("Error getting the clap vector from: %s", song_id)
                            logger.error(e)
                            continue
                        clap_bytes = clap.cpu().numpy().astype(np.float32).tobytes()
                        clap_length = len(clap_bytes)
                        assert clap_length == clap.size(0) * clap.size(1) * 4
                        clap_end = clap_offset + clap_length
                        try:
                            clap_dat_file.write(clap_bytes)
                        except RuntimeError as e:
                            logger.error(f"Failed to write Clap data into packed file.")
                            logger.error(e)
                            exit(4)
                        try:
                            clap_csv_writer.writerow([dataset, song_id, clap_offset, clap_end, clap_length])
                            clap_offset = clap_end
                        except RuntimeError as e:
                            logger.error(f"Failed to update the Clap CSV file.")
                            logger.error(e)
                            exit(4)
    logger.info("All Done.")

if __name__ == "__main__":
    main()