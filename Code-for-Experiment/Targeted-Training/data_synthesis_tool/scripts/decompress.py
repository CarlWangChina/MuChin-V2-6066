import soundfile as sf
import pyloudnorm as pyln
import librosa
import numpy
import sys
import os
from concurrent.futures import ProcessPoolExecutor

def process_file(path_in:str, path_out:str):
    try:
        data, rate = sf.read(path_in)
        meter = pyln.Meter(rate)
        loudness = meter.integrated_loudness(data)
        data = pyln.normalize.loudness(data, loudness, -12.0)
        data = librosa.resample(data.T, orig_sr=rate, target_sr=44100).T
        if data.ndim == 1:
            data = numpy.dstack((data, data)).squeeze()
        sf.write(path_out, data, 44100)
    except Exception as err:
        print(err)

def process_path_list(file_list_path: str, input_dir: str, output_dir: str):
    with open(file_list_path, 'r') as f:
        file_list = [line.strip() for line in f]
    with ProcessPoolExecutor() as executor:
        futures = []
        for filename in file_list:
            input_file_path = os.path.join(input_dir, filename)
            base_filename = os.path.splitext(filename)[0]
            output_file_path = os.path.join(output_dir, base_filename + '.wav')
            if os.path.exists(output_file_path):
                continue
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            future = executor.submit(process_file, input_file_path, output_file_path)
            futures.append(future)
        for future in futures:
            future.result()

process_path_list("/nfs/data/datasets-wav/index/bandcamp_0.txt", "/nfs/data/datasets-mp3/", "/nfs/data/datasets-wav/datas/")
process_path_list("/nfs/data/datasets-wav/index/bandcamp_1.txt", "/nfs/data/datasets-mp3/", "/nfs/data/datasets-wav/datas/")
process_path_list("/nfs/data/datasets-wav/index/bandcamp_2.txt", "/nfs/data/datasets-mp3/", "/nfs/data/datasets-wav/datas/")
process_path_list("/nfs/data/datasets-wav/index/bandcamp_3.txt", "/nfs/data/datasets-mp3/", "/nfs/data/datasets-wav/datas/")
process_path_list("/nfs/data/datasets-wav/index/bandcamp_4.txt", "/nfs/data/datasets-mp3/", "/nfs/data/datasets-wav/datas/")
process_path_list("/nfs/data/datasets-wav/index/bandcamp_5.txt", "/nfs/data/datasets-mp3/", "/nfs/data/datasets-wav/datas/")
process_path_list("/nfs/data/datasets-wav/index/bandcamp_6.txt", "/nfs/data/datasets-mp3/", "/nfs/data/datasets-wav/datas/")
process_path_list("/nfs/data/datasets-wav/index/bandcamp_7.txt", "/nfs/data/datasets-mp3/", "/nfs/data/datasets-wav/datas/")