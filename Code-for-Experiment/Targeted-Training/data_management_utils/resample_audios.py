import concurrent.futures
import torchaudio
import random
import csv
import urllib.parse
import pyloudnorm as pyln
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path

INPUT_DIRS = {
    "ali": Path("/mnt/d/Data/Music/dataset-mp3/ali"),
    "cb": Path("/mnt/d/Data/Music/dataset-mp3/cb"),
    "dyqy": Path("/mnt/d/Data/Music/dataset-mp3/dyqy"),
    "mysong": Path("/mnt/nfs/mysong_all_10000")
}
OUTPUT_DIR = Path("/mnt/nfs/packed-wav-24000-mono")
MAX_FILES_PER_SUBDIR = 180000
NORMALIZED_LOUDNESS = -14.0
assert OUTPUT_DIR.is_dir()

wav_lists = [[] for _ in range(8)]
for dataset, input_dir in INPUT_DIRS.items():
    file_count = 0
    for file in tqdm(input_dir.glob("**/*.mp3"), desc=f"Listing files from {dataset}"):
        list_id = random.randint(0, 7)
        wav_lists[list_id].append(file)
        file_count += 1
        if file_count >= MAX_FILES_PER_SUBDIR:
            break

for lst in wav_lists:
    random.shuffle(lst)

total_files = sum(len(wav_list) for wav_list in wav_lists)
print("Number of files in each list:")
for i, wav_list in enumerate(wav_lists):
    print(f"List {i}: {len(wav_list)}")
print("Total files:", total_files)

def get_song_id(file: Path) -> (str, str):
    for dataset, input_dir in INPUT_DIRS.items():
        if str(file).startswith(str(input_dir)):
            song_id = str(file.relative_to(input_dir.parent))
            if song_id.endswith("_src.mp3"):
                song_id = song_id[:-len("_src.mp3")]
            elif song_id.endswith(".mp3"):
                song_id = song_id[:-len(".mp3")]
            song_id = urllib.parse.quote(song_id)
            return song_id, dataset
    assert False, f"File {file} not found in any dataset"

def resample_group_task(list_id: int):
    wav_list = wav_lists[list_id]
    output_csv_file = OUTPUT_DIR / f"wav_list_{list_id}.csv"
    output_data_file = OUTPUT_DIR / f"wav_list_{list_id}.dat"
    meter = pyln.Meter(24000)
    with open(output_csv_file, "w", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(["dataset", "song_id", "offset", "end", "length"])
        with open(output_data_file, "wb") as data_f:
            offset = 0
            if list_id == 0:
                pbar = tqdm(total=len(wav_list), desc=f"Resampling")
            else:
                pbar = None
            for file in wav_list:
                song_id, dataset = get_song_id(file)
                try:
                    audio, sr = torchaudio.load(file)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
                    continue
                try:
                    resampler = torchaudio.transforms.Resample(sr, 24000)
                    resampled_audio = resampler(audio).transpose(0, 1).numpy()
                    loudness = meter.integrated_loudness(resampled_audio)
                    resampled_audio = pyln.normalize.loudness(resampled_audio, loudness, NORMALIZED_LOUDNESS)
                    resampled_audio = resampled_audio.mean(axis=1, keepdims=False)
                    resampled_audio = np.clip(resampled_audio * 32768.0, a_min=-32768, a_max=32767).astype(np.int16)
                    buffer = resampled_audio.tobytes()
                    length = len(buffer)
                    writer.writerow([dataset, song_id, offset, offset + length, length])
                    data_f.write(buffer)
                    data_f.flush()
                    f.flush()
                    end = data_f.tell()
                    assert end == offset + length, "File length mismatch.  Disk full?"
                    offset = end
                except Exception as e:
                    print(f"Error processing {file}: {e}")
                    continue
                if pbar is not None:
                    pbar.update(1)
            if pbar is not None:
                pbar.close()
    return list_id

from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(resample_group_task, list_id) for list_id in range(8)]
    concurrent.futures.wait(futures)
    for future in concurrent.futures.as_completed(futures):
        print("Task completed ", future.result())