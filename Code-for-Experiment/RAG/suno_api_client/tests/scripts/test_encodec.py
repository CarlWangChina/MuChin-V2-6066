import os
import sys
import torchaudio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sunoapi.encodec import *
import sunoapi.config_loader as config

processor = EncoedecProcessor()
audio_path = config.PROJECT_ROOT + "/tests/outputs/test_gen/"
os.makedirs(config.PROJECT_ROOT + "/tests/outputs/test_processed/", exist_ok=True)

for root, dirs, files in os.walk(audio_path):
    for file in files:
        if file.endswith(".mp3"):
            file_path = os.path.join(root, file)
            audio_data, sr = torchaudio.load(file_path)
            processed_data, nsr = processor.process(audio_data, sr)
            print(audio_data.shape, processed_data.shape, sr, nsr)
            torchaudio.save(config.PROJECT_ROOT + "/tests/outputs/test_processed/" + file, processed_data, nsr)