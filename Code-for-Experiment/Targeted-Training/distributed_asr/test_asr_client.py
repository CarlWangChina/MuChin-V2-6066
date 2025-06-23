import requests
import json

url = 'http://localhost:8000/ASRDec'
audio_path = 'data\\test_audio.wav'

with open(audio_path, 'rb') as audio_file:
    audio_bytes = audio_file.read()

response = requests.post(url, data=audio_bytes)

if response.status_code == 200:
    result = response.json()
    with open('transcription_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
else:
    print(f"Request failed with status code {response.status_code}")