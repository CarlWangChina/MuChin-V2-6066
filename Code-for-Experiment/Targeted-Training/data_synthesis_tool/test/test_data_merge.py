import os
import sys
import csv
import wave
import urllib.parse
import hashlib

def urldecode(encoded_string):
    decoded_string = urllib.parse.unquote(encoded_string)
    return decoded_string

def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

class WavExtractor:
    def __init__(self, bin_file_path, csv_file_path):
        self.bin_file_path = bin_file_path
        self.csv_file_path = csv_file_path
        self.meta_data = []

    def load_meta_data(self):
        with open(self.csv_file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)
            for row in csv_reader:
                self.meta_data.append({
                    "path": row[1],
                    "offset_begin": int(row[2]),
                    "offset_end": int(row[3]),
                    "length": int(row[4])
                })

    def extract_wav(self, output_file_path, search_file):
        with open(self.bin_file_path, 'rb') as bin_file:
            with wave.open(output_file_path, 'wb') as wav_out:
                for meta in self.meta_data:
                    if meta["path"] == search_file:
                        bin_file.seek(meta["offset_begin"])
                        frames = bin_file.read(meta["length"])
                        wav_out.setparams((2, 2, 44100, 0, 'NONE', 'not compressed'))
                        wav_out.writeframes(frames)
                        break

wav_extractor = WavExtractor("/nfs/data/datasets-wav/merged/4-tmp/audios.bin", "/nfs/data/datasets-wav/merged/4-tmp/audio_metas.csv")
wav_extractor.load_meta_data()
count = 0
with open("/nfs/data/datasets-wav/merged/4-tmp/audio_metas.csv", 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    for row in csv_reader:
        if count % 10 == 0:
            songid = row[1]
            filename = "/nfs/data/datasets-wav/data/" + row[0] + "/" + urldecode(row[1])
            if os.path.exists(filename + ".wav"):
                filename = filename + ".wav"
            elif os.path.exists(filename + "_src.wav"):
                filename = filename + "_src.wav"
            elif os.path.exists("/nfs/data/datasets-wav/data/" + row[0] + "/" + row[1][:2] + "/" + urldecode(row[1]) + "_src.wav"):
                filename = "/nfs/data/datasets-wav/data/" + row[0] + "/" + row[1][:2] + "/" + urldecode(row[1]) + "_src.wav"
            else:
                print(f"File not found: {filename}")
                continue
            wav_extractor.extract_wav("/tmp/test_data_merge.wav", songid)
            extracted_md5 = calculate_md5("/tmp/test_data_merge.wav")
            original_md5 = calculate_md5(filename)
            print(f"Original filename: {filename} Extracted MD5: {extracted_md5} Original MD5: {original_md5}")
            if extracted_md5 == original_md5:
                pass
            else:
                print("MD5 mismatch: The extracted file is different from the original file.")
        count += 1