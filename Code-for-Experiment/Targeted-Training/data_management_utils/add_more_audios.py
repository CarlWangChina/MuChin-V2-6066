import torchaudio
from pathlib import Path
from tqdm.auto import tqdm
import random
import csv
import pyloudnorm as pyln
import numpy as np

PACKED_DIR = Path("/nfs/data/datasets-wav-24000-mono")
NEW_SONGS_DIR = Path("/nfs/data/datasets-mp3/ali-40w-extra")
MAX_DATASET_SIZE = 53000
NORMALIZED_LOUDNESS = -14.0
random.seed(888)
song_id_set = set()
song_list = []
for i in range(8):
    meta_file = PACKED_DIR / f"wav_list_{i}.csv"
    data_file = PACKED_DIR / f"wav_list_{i}.dat"
    song_list.append([])
    end_pos = 0
    with open(meta_file, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            song_id = row[1]
            song_id_set.add(song_id)
            song_list[i].append(song_id)
            end_pos = int(row[3])
    with open(data_file, "r") as f:
        file_size = f.seek(0, 2)
        assert file_size == end_pos, f"File size mismatch for list {i}: {file_size} != {end_pos}"
    print(f"List {i}: {len(song_list[i])} songs")
print("Appending new songs...")
new_song_list = []
for file in NEW_SONGS_DIR.glob("**/*.mp3"):
    song_id = "ali/65/" + file.stem
    if song_id not in song_id_set:
        new_song_list.append((song_id, file))
print(f"Found {len(new_song_list)} new songs")
random.shuffle(new_song_list)
song_index = 0
meter = pyln.Meter(24000)

def append_song(song_id, filename, csv_writer, csv_file, data_file):
    audio, sr = torchaudio.load(filename)
    if audio.size(-1) < 24000 or audio.size(1) > 24000 * 600:
        raise ValueError("Audio too short or too long")
    resampler = torchaudio.transforms.Resample(sr, 24000)
    resampled_audio = resampler(audio).transpose(0, 1).numpy()
    loudness = meter.integrated_loudness(resampled_audio)
    resampled_audio = pyln.normalize.loudness(resampled_audio, loudness, NORMALIZED_LOUDNESS)
    resampled_audio = resampled_audio.mean(axis=1, keepdims=False)
    resampled_audio = np.clip(resampled_audio * 32768.0, a_min=-32768, a_max=32767).astype(np.int16)
    buffer = resampled_audio.tobytes()
    length = len(buffer)
    offset = data_file.tell()
    csv_writer.writerow(["ali2", song_id, offset, offset + length, length])
    data_file.write(buffer)
    data_file.flush()
    csv_file.flush()
    end = data_file.tell()
    assert end == offset + length, "File length mismatch.  Disk full?"

for i in range(8):
    meta_file = PACKED_DIR / f"wav_list_{i}.csv"
    data_file = PACKED_DIR / f"wav_list_{i}.dat"
    if len(song_list[i]) >= MAX_DATASET_SIZE:
        continue
    num_songs = MAX_DATASET_SIZE - len(song_list[i])
    with open(meta_file, "a") as meta_file:
        csv_writer = csv.writer(meta_file)
        with open(data_file, "ab") as data_file:
            with tqdm(total=num_songs, desc=f"Appending list {i}") as pbar:
                num_songs_processed = 0
                while num_songs_processed < num_songs:
                    song_id, filename = new_song_list[song_index]
                    song_index += 1
                    try:
                        append_song(song_id, filename, csv_writer=csv_writer, csv_file=meta_file, data_file=data_file)
                        song_list[i].append(song_id)
                        pbar.update(1)
                        num_songs_processed += 1
                    except Exception as e:
                        print(f"Error appending song {song_id}: {e}")