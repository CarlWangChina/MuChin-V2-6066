import csv
import numpy as np
import sqlite3
from pathlib import Path
from tqdm.auto import tqdm
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger

logger = get_logger(__name__)
NUM_NODES = 8
PACKED_DB_FILE = Path("/mnt/d/Data/Music/packed-lyrics.sqlite3")
WAV_LIST_DIR = Path("/mnt/d/Data/Music/wav-lists")
OUTPUT_DIR = Path("/mnt/d/Data/Music/packed_lyric_tokens")

def main():
    assert WAV_LIST_DIR.is_dir(), f"Directory {WAV_LIST_DIR} does not exist."
    assert OUTPUT_DIR.is_dir(), f"Directory {OUTPUT_DIR} does not exist."
    conn = sqlite3.connect(PACKED_DB_FILE)
    cursor = conn.cursor()
    for i in range(NUM_NODES):
        wav_list_file = WAV_LIST_DIR / f"wav_list_{i}.csv"
        lyric_list_file = OUTPUT_DIR / f"lyric_list_{i}.csv"
        lyric_data_file = OUTPUT_DIR / f"lyric_list_{i}.dat"
        logger.info("Writing lyric tokens for node %d", i)
        with open(wav_list_file, "r") as f:
            reader = csv.reader(f)
            next(reader)
            wav_list = list(reader)
        with open(lyric_list_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["dataset", "song_id", "offset", "end", "length"])
            offset = 0
            with open(lyric_data_file, "wb") as data_f:
                for wav_list_item in tqdm(wav_list, desc=f"Node {i}"):
                    dataset = wav_list_item[0]
                    song_id = wav_list_item[1]
                    cursor.execute(f"SELECT tokens FROM lyrics WHERE song_id = ?", (song_id,))
                    tokens = cursor.fetchone()[0]
                    if tokens is None:
                        continue
                    tokens = tokens.split()
                    tokens = np.array(tokens, dtype=np.int32)
                    token_bytes = tokens.tobytes()
                    length = len(token_bytes)
                    end = offset + length
                    data_f.write(token_bytes)
                    writer.writerow([dataset, song_id, offset, end, length])
                    offset = end
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()