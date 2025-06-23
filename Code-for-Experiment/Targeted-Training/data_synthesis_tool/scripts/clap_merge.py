import wave
import urllib.parse
import hashlib
import csv
import os
import torch
import numpy
import multiprocessing

def urlencode(input_string):
    input_bytes = input_string.encode()
    encoded_string = urllib.parse.quote(input_bytes)
    return encoded_string

def pt2ArrBuffer(path: str):
    x = torch.load(path, map_location='cpu')
    assert torch.is_tensor(x), f"{x}"
    assert x.dim() == 3, f"{x.dim()}"
    assert x.shape[2] == 512, f"{x.shape}"
    assert x.shape[1] == 1 or x.shape[1] == 2, f"{x.shape}"
    x = x.mean(dim=1).flatten().cpu().type(torch.float32).numpy().tobytes()
    return x

class ClapMerge:
    def __init__(self, out_csv, out_bin):
        self.out_csv = out_csv
        self.out_bin = out_bin
        if os.path.exists(out_bin):
            self.current_length = os.path.getsize(out_bin)
        else:
            self.current_length = 0
        self.clap_meta_data = open(out_csv, "a", newline='')
        self.clap_meta_writer = csv.writer(self.clap_meta_data)
        if self.current_length == 0:
            self.clap_meta_writer.writerow(["songid", "offset_begin", "offset_end", "length"])
            print("create header")
        self.clap_datas_output = open(out_bin, "ab")

    def merge_pt(self, input_file, dataset, songid):
        frames = pt2ArrBuffer(input_file)
        data_len = len(frames)
        self.clap_datas_output.write(frames)
        self.clap_meta_writer.writerow([dataset+"/"+songid, self.current_length, self.current_length + data_len - 1, data_len])
        self.current_length += data_len

    def process_list(self, path):
        with open(path, 'r') as f:
            line = f.readline()
            while line:
                path = line.strip()
                arr = path.split("/")
                dataset = arr[0]
                songid = "/".join(arr[1:]).replace("_src.wav", "").replace("_src.mp3", "")
                input_file = "/nfs/data/clap/vecs/" + path.replace(".mp3", ".pt")
                if os.path.exists(input_file):
                    print(path,":",input_file)
                    self.merge_pt(input_file, dataset, songid)
                line = f.readline()
        return self

    def process_list2(self, path):
        with open(path, 'r') as f:
            line = f.readline()
            while line:
                path = line.strip()
                arr = path.split("/")
                dataset = arr[0]
                songid = urlencode("/".join(arr[1:]).replace(".mp3", ""))
                input_file = "/nfs/data/clap/vecs/" + path.replace(".mp3", ".pt")
                if os.path.exists(input_file):
                    print(path,":",input_file)
                    self.merge_pt(input_file, dataset, songid)
                line = f.readline()
        return self

if __name__ == "__main__":
    for i in range(0,8):
        print("merge:",i)
        ClapMerge(f"/nfs/data/clap/merged-fix/{i}/clap_metas_fix.csv", f"/nfs/data/clap/merged-fix/{i}/claps.bin").process_list(f"/nfs/data/datasets-wav/index/common_{i}.txt").process_list2(f"/nfs/data/datasets-wav/index/bandcamp_{i}.txt")