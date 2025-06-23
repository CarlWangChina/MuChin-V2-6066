import requests
import json
import os
import torch
import torchaudio
import pyloudnorm as pyln

FILE_DIR = os.path.dirname(__file__)
url = 'http://127.0.0.1:8000/ASR'
audio_path = os.path.join(FILE_DIR, '..', 'data', 'test_audio.mp3')

with open(audio_path, "rb") as fp:
    data = fp.read()
    print(len(data))
    response = requests.post(url, data=data)
    if response.status_code == 200:
        result = response.text
        print("result:", result)
    else:
        print(f"Request failed with status code {response.status_code}")