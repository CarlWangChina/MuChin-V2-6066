import os
import torch
from transformers import EncodecModel, EncodecFeatureExtractor

device = "cuda:7"
model = EncodecModel.from_pretrained("facebook/encodec_48khz").to(device)
processor = EncodecFeatureExtractor.from_pretrained("facebook/encodec_48khz")

def process_file(path):
    data = torch.load(path, map_location=device)["audio_codes"]
    frames = [model.quantizer.decode(frame).reshape(8, -1) for frame in data]
    frames = torch.cat(frames)
    mean = frames.mean(dim=0, keepdim=True)
    print(path)
    torch.save(mean, path + ".mean")

dir_path = '/nfs/music-5-test/encodec/encode/'
for dirpath, dirnames, filenames in os.walk(dir_path):
    for filename in filenames:
        file_path = os.path.join(dirpath, filename)
        if file_path.endswith('.pkl'):
            process_file(file_path)