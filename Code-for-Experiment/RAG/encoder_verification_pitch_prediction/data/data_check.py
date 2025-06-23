import os
from tqdm import tqdm

with open("data/Code-for-Experiment/Targeted-Training/token_transformer_module/scripts/acoustic_tokenize/data_control_files/music-type-checked-1087.txt", "r") as f:
    lines1 = f.readlines()

with open("data/Code-for-Experiment/Targeted-Training/token_transformer_module/scripts/acoustic_tokenize/data_control_files/music-type-checked.txt", "r") as f:
    lines2 = f.readlines()

lines = lines1 + lines2
songids = []

for line in tqdm(lines):
    data_dict = eval(line)
    path = data_dict["path"]
    songid = os.path.basename(path)
    songids.append(songid)

data_list = os.listdir("data/midi")
existing_list = []

for filename in tqdm(data_list):
    songid = filename.split(".")[0]
    if songid in songids:
        existing_list.append(f"{songid}\n")

with open("data/Code-for-Experiment/RAG/encoder_verification_pitch_prediction/data/existing_data.txt", "w") as f:
    f.writelines(existing_list)