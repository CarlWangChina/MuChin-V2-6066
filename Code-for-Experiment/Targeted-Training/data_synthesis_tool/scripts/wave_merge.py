import wave
import urllib.parse
import hashlib
import csv
import os

def urlencode(input_string):
    input_bytes = input_string.encode()
    encoded_string = urllib.parse.quote(input_bytes)
    return encoded_string

class WaveMerge:
    def __init__(self, out_csv, out_bin):
        self.out_csv = out_csv
        self.out_bin = out_bin
        if os.path.exists(out_bin):
            self.current_length = os.path.getsize(out_bin)
        else:
            self.current_length = 0
        self.audio_meta_data = open(out_csv, "a", newline='')
        self.audio_meta_writer = csv.writer(self.audio_meta_data)
        if self.current_length == 0:
            self.audio_meta_writer.writerow(["dataset", "songid", "offset_begin", "offset_end", "length"])
            print("create header")
        self.audio_datas_output = open(out_bin, "ab")

    def merge_wav(self, input_file, dataset, songid):
        with wave.open(input_file, 'rb') as wav_in:
            params = wav_in.getparams()
            frames = wav_in.readframes(-1)
            assert params.nchannels == 2
            assert params.sampwidth == 2
            assert params.framerate == 44100
            data_len = len(frames)
            self.audio_datas_output.write(frames)
            self.audio_meta_writer.writerow([dataset, songid, self.current_length, self.current_length + data_len - 1, data_len])
            self.current_length += data_len

    def process_list(self, path):
        with open(path, 'r') as f:
            line = f.readline()
            while line:
                path = line.strip()
                arr = path.split("/")
                dataset = arr[0]
                songid = arr[-1].replace("_src.wav", "").replace("_src.mp3", "")
                input_file = "/nfs/data/datasets-wav/data/" + path.replace(".mp3", ".wav")
                if os.path.exists(input_file):
                    print(input_file)
                    self.merge_wav(input_file, dataset, songid)
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
                input_file = "/nfs/data/datasets-wav/data/" + path.replace(".mp3", ".wav")
                if os.path.exists(input_file):
                    print(input_file)
                    self.merge_wav(input_file, dataset, songid)
                line = f.readline()
        return self

WaveMerge("/nfs/data/datasets-wav/merged/4-tmp/audio_metas.csv","/nfs/data/datasets-wav/merged/4-tmp/audios.bin").process_list("/nfs/data/datasets-wav/index/common_4.txt").process_list2("/nfs/data/datasets-wav/index/bandcamp_4.txt")