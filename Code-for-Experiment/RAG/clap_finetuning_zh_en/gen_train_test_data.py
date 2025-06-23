import os
import json
import random
import re

def merge_strings(strings, max_length, min_length):
    result = []
    current_str = ""
    for string in strings:
        if len(current_str) + len(string) > max_length:
            result.append(current_str)
            current_str = string
        else:
            current_str += " " + string
    if current_str:
        result.append(current_str)
    result = [r for r in result if min_length <= len(r) <= max_length]
    return result

def clean_string(s):
    return re.sub(r'[^a-zA-Z0-9 ]', '', s.lower())

def get_audio_path(audio_path, split_audio_path):
    audio_id = audio_path.split("/")[-2]
    short_audio_path = os.path.join(split_audio_path, audio_id, "short")
    medium_audio_path = os.path.join(split_audio_path, audio_id, "medium")
    long_audio_path = os.path.join(split_audio_path, audio_id, "long")
    if os.path.exists(short_audio_path):
        short_audio_paths = [os.path.join(short_audio_path, file) for file in os.listdir(short_audio_path)]
    else:
        short_audio_paths = []
    if os.path.exists(medium_audio_path):
        medium_audio_paths = [os.path.join(medium_audio_path, file) for file in os.listdir(medium_audio_path)]
    else:
        medium_audio_paths = []
    if os.path.exists(long_audio_path):
        long_audio_paths = [os.path.join(long_audio_path, file) for file in os.listdir(long_audio_path)]
    else:
        long_audio_paths = []
    return short_audio_paths, medium_audio_paths, long_audio_paths

def generate_desc(descs):
    descs = [clean_string(s) for s in descs if s is not None]
    descs = [s for s in descs if s is not None]
    descs = sorted(descs, key=len)
    short_desc = merge_strings(descs, 30, 0)
    medium_desc = merge_strings(descs, 100, 30)
    long_desc = merge_strings(descs, 240, 100)
    return short_desc, medium_desc, long_desc

def _generate_examples(data_dir):
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        with open(file_path, "r") as f:
            file_json = json.load(f)
        audio_path = file_json['song_dir']
        if not os.path.exists(audio_path):
            continue
        short_audio_paths, medium_audio_paths, long_audio_paths = get_audio_path(audio_path, "muchin_dataset_cut")
        all_short_descs, all_medium_descs, all_long_descs = [], [], []
        if file_json['amat'] is not None:
            short_amat_descs, medium_amat_descs, long_amat_descs = generate_desc(file_json['amat'])
            all_short_descs += short_amat_descs
            all_medium_descs += medium_amat_descs
            all_long_descs += long_amat_descs
        if file_json['prof'] is not None:
            short_prof_descs, medium_prof_descs, long_prof_descs = generate_desc(file_json['prof'])
            all_short_descs += short_prof_descs
            all_medium_descs += medium_prof_descs
            all_long_descs += long_prof_descs
        all_short_descs = random.sample(all_short_descs, 24) if len(all_short_descs) >= 24 else all_short_descs
        all_medium_descs = random.sample(all_medium_descs, 24) if len(all_medium_descs) >= 24 else all_medium_descs
        all_long_descs = random.sample(all_long_descs, 24) if len(all_long_descs) >= 24 else all_long_descs
        idx = 0
        for descs in [all_short_descs, all_medium_descs, all_long_descs]:
            random_short_audio_paths = random.sample(short_audio_paths, 8) if len(short_audio_paths) >= 8 else short_audio_paths
            random_medium_audio_paths = random.sample(medium_audio_paths, 8) if len(medium_audio_paths) >= 8 else medium_audio_paths
            random_long_audio_paths = random.sample(long_audio_paths, 8) if len(long_audio_paths) >= 8 else long_audio_paths
            audio_paths = random_short_audio_paths + random_medium_audio_paths + random_long_audio_paths
            for i in range(min(len(audio_paths), len(descs))):
                audio_name = os.path.basename(audio_path).replace(".", f"_{idx}.")
                with open(f"{os.path.basename(data_dir)}.txt", "a") as f:
                    f.write(f"{audio_name} {audio_paths[i]} {descs[i]}\n")
                idx += 1

_generate_examples("processed_datasets/muchin_en/train")
_generate_examples("processed_datasets/muchin_en/test")