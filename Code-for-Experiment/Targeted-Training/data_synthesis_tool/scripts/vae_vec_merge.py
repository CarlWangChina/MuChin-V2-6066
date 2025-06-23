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
    pt = torch.load(path)
    n = pt[0].shape[0]
    x = pt[0].reshape(n, 150, 128).contiguous()
    x = x.reshape(n * 150 * 128)
    x_bytes = x.numpy().astype(numpy.float32).tobytes()
    return x_bytes

class VaeMerge:
    def __init__(self, out_csv, out_bin):
        self.out_csv = out_csv
        self.out_bin = out_bin
        if os.path.exists(out_bin):
            self.current_length = os.path.getsize(out_bin)
        else:
            self.current_length = 0
        self.vae_meta_data = open(out_csv, "a", newline='')
        self.vae_meta_writer = csv.writer(self.vae_meta_data)
        if self.current_length == 0:
            self.vae_meta_writer.writerow(["dataset", "songid", "offset_begin", "offset_end", "length"])
            print("create header")
        self.vae_datas_output = open(out_bin, "ab")

    def merge_pt(self, input_file, dataset, songid):
        frames = pt2ArrBuffer(input_file)
        data_len = len(frames)
        self.vae_datas_output.write(frames)
        self.vae_meta_writer.writerow([dataset, songid, self.current_length, self.current_length + data_len - 1, data_len])
        self.current_length += data_len

    def process_list(self, path):
        with open(path, 'r') as f:
            line = f.readline()
            while line:
                path = line.strip()
                arr = path.split("/")
                dataset = arr[0]
                songid = arr[-1].replace("_src.wav", "").replace("_src.mp3", "")
                input_file = "/nfs/data/encodec-vae/vecs/" + path.replace(".mp3", ".pt")
                if os.path.exists(input_file):
                    print(input_file)
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
                input_file = "/nfs/data/encodec-vae/vecs/" + path.replace(".mp3", ".pt")
                if os.path.exists(input_file):
                    print(input_file)
                    self.merge_pt(input_file, dataset, songid)
                line = f.readline()
        return self

def process_task(process_num):
    print(f"process{process_num+1}")
    VaeMerge(f"/nfs/data/encodec-vae/merged/{process_num}/vae_metas.csv", f"/nfs/data/encodec-vae/merged/{process_num}/vae.bin").process_list(f"/nfs/data/datasets-wav/index/common_{process_num}.txt").process_list2(f"/nfs/data/datasets-wav/index/bandcamp_{process_num}.txt")

if __name__ == "__main__":
    processes = []
    for i in range(8):
        p = multiprocessing.Process(target=process_task, args=(i,))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()