import os
from pydub import AudioSegment
import random
from tqdm import tqdm
from multiprocessing import Pool

def split_mp3_random(file_path, output_dir, min_duration=10000, max_duration=30000):
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) != 0:
        return
    try:
        audio = AudioSegment.from_mp3(file_path)
        audio_length = len(audio)
        current_position = 0
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        i = 0
        while current_position < audio_length:
            chunk_length = random.randint(min_duration, max_duration)
            if current_position + chunk_length > audio_length:
                chunk_length = audio_length - current_position
            chunk = audio[current_position:current_position + chunk_length]
            if len(chunk) >= min_duration:
                chunk_name = f"{output_dir}/{os.path.basename(file_path).replace('.mp3', '')}_{i+1}.mp3"
                chunk.export(chunk_name, format="mp3")
                i += 1
            current_position += chunk_length
    except Exception as e:
        os.system(f"rm -rf {output_dir}")

audio_path = [os.path.join("muchin_dataset/muchin", file, f"{file}_src.mp3") for file in os.listdir("muchin_dataset/muchin") if os.path.exists(os.path.join("muchin_dataset/muchin", file, f"{file}_src.mp3"))]
short_audio_output_path = ["/".join(audio_p.split("/")[:3]).replace("muchin_dataset", "muchin_dataset_cut").replace("muchin/", "") + "/short" for audio_p in audio_path]
medium_audio_output_path = ["/".join(audio_p.split("/")[:3]).replace("muchin_dataset", "muchin_dataset_cut").replace("muchin/", "") + "/medium" for audio_p in audio_path]
long_audio_output_path = ["/".join(audio_p.split("/")[:3]).replace("muchin_dataset", "muchin_dataset_cut").replace("muchin/", "") + "/long" for audio_p in audio_path]

with Pool() as pool:
    futures = [
        pool.apply_async(split_mp3_random, (audio_p, short_audio_output_p, 5000, 15000))
        for audio_p, short_audio_output_p in zip(audio_path, short_audio_output_path)
    ]
    _ = [future.get() for future in tqdm(futures)]

with Pool() as pool:
    futures = [
        pool.apply_async(split_mp3_random, (audio_p, medium_audio_output_p, 15000, 25000))
        for audio_p, medium_audio_output_p in zip(audio_path, medium_audio_output_path)
    ]
    _ = [future.get() for future in tqdm(futures)]

with Pool() as pool:
    futures = [
        pool.apply_async(split_mp3_random, (audio_p, long_audio_output_p, 25000, 35000))
        for audio_p, long_audio_output_p in zip(audio_path, long_audio_output_path)
    ]
    _ = [future.get() for future in tqdm(futures)]