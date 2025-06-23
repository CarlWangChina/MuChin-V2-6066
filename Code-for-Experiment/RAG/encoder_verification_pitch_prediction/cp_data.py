import os
import shutil
from tqdm import tqdm

mert_root_path = "/nfs/mert/mert-v1-330m-75hz/"
midi_path = "data/midi"
dst_path = "/data/xary/mert"

for i in range(6):
    mert_path = os.path.join(mert_root_path, f"{i}")
    dirlist = os.listdir(mert_path)
    for filename in tqdm(dirlist):
        midi_file = filename.split("_")[0] + '.mp3_5b.mid'
        midi_file_path = os.path.join(midi_path, midi_file)
        if not os.path.exists(midi_file_path):
            continue
        check_path = os.path.join(dst_path, filename)
        if os.path.exists(check_path):
            continue
        mert = os.path.join(mert_path, filename)
        shutil.copy(mert, dst_path)