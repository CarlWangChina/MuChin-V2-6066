import os
import sys
import shutil

def process_path_list(source_file, source_prefix, target_prefix):
    os.makedirs(target_prefix, exist_ok=True)
    with open(source_file, 'r') as f_source:
        for line in f_source:
            line = line.strip()
            source_path = os.path.join(source_prefix, line).replace(".mp3", ".wav")
            target_path = os.path.join(target_prefix, line).replace(".mp3", ".wav")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            print(source_path, target_path)
            try:
                shutil.move(source_path, target_path)
            except Exception:
                pass

process_path_list("/nfs/data/datasets-wav/index/common_1.txt", "/nfs/data/datasets-wav/datas/", "/data/datasets-wav/datas/")