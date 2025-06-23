import os
import json

def traverse_folders(root_folder):
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith(".pt"):
                yield file_path

with open('encode_list.json', 'w') as f:
    for file_path in traverse_folders('/export/data/clap/encode/audio/'):
        objname = os.path.basename(file_path)
        audioname1 = file_path.replace(".pt", ".mp3").replace("/export/data/clap/encode/audio/", "/export/data/datasets-mp3/")
        audioname2 = file_path.replace(".pt", ".wav").replace("/export/data/clap/encode/audio/", "/export/data/datasets-mp3/")
        if os.path.exists(audioname1):
            f.write(json.dumps({"name": objname, "path": audioname1}) + "\n")
        elif os.path.exists(audioname2):
            f.write(json.dumps({"name": objname, "path": audioname2}) + "\n")
        else:
            print(f"{audioname1} or {audioname2} does not exist")
    for file_path in traverse_folders('/export/data/lrc-vec/encode/'):
        objname = os.path.basename(file_path).replace(".pt","")
        audioname1 = file_path.replace(".pt", "_src.mp3").replace("/export/data/lrc-vec/encode/", "/export/data/datasets-mp3/cb/")
        audioname2 = file_path.replace(".pt", "_src.wav").replace("/export/data/lrc-vec/encode/", "/export/data/datasets-mp3/cb/")
        if os.path.exists(audioname1):
            f.write(json.dumps({"name": objname, "path": audioname1}) + "\n")
        elif os.path.exists(audioname2):
            f.write(json.dumps({"name": objname, "path": audioname2}) + "\n")
        else:
            print(f"{audioname1} or {audioname2} does not exist")