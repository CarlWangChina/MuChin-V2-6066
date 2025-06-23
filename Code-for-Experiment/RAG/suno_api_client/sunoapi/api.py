import requests
import time
import subprocess
import traceback
import os
from tqdm import tqdm
import sunoapi.config_loader as config
from requests.exceptions import RequestException, ConnectTimeout, ReadTimeout
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

api_base_url = "https://api.sunoaiapi.com/api/v1"
headers = {
    "api-key": config.SUNO_API_KEY,
    "Content-Type": "application/json"
}

def concat_audio(clip_id):
    url = f"{api_base_url}/gateway/generate/concat"
    data = {
        "clip_id": clip_id
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def check_account_limit():
    url = f"{api_base_url}/gateway/limit"
    response = requests.get(url, headers=headers)
    return response.json()

def generate_music(title, tags, prompt, mv="chirp-v3-5", continue_at=None, continue_clip_id=None):
    url = f"{api_base_url}/gateway/generate/music"
    data = {
        "title": title,
        "tags": tags,
        "prompt": prompt,
        "mv": mv,
    }
    if continue_at is not None:
        data["continue_at"] = continue_at
    if continue_clip_id is not None:
        data["continue_clip_id"] = continue_clip_id
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def check_generation_result(song_id):
    url = f"{api_base_url}/gateway/feed/{song_id}"
    response = requests.get(url, headers=headers)
    return response.json()

def generate_music_by_description(description_prompt, make_instrumental=False):
    url = f"{api_base_url}/gateway/generate/gpt_desc"
    data = {
        "gpt_description_prompt": description_prompt,
        "make_instrumental": make_instrumental,
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def generate_lyrics(prompt):
    url = f"{api_base_url}/gateway/generate/lyrics"
    data = {
        "prompt": prompt
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def check_lyrics_result(lid):
    url = f"{api_base_url}/gateway/lyrics/{lid}"
    response = requests.get(url, headers=headers)
    return response.json()

def download_file_with_retry(url, file_name, max_retries=30, retry_delay=30):
    logger.info("download %s->%s", file_name, url)
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            with open(file_name, 'wb') as file:
                for data in response.iter_content(block_size):
                    if data:
                        file.write(data)
            logger.info("download %s success", file_name)
            return
        except (RequestException, ConnectTimeout, ReadTimeout) as e:
            retries += 1
            if retries >= max_retries:
                logger.error("download %s->%s failed: %s", file_name, url, e)
                return
            logger.info("download %s->%s failed: %s, retrying...", file_name, url, e)
            time.sleep(retry_delay)

def download_file_with_progress(url, file_name):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(file_name, 'wb') as file:
        for data in response.iter_content(block_size):
            t.update(len(data))
            file.write(data)
    t.close()

def download_result(music_result, outdir="outputs"):
    os.makedirs(outdir, exist_ok=True)
    res = []
    for song in music_result['data']:
        song_id = song['song_id']
        while True:
            generation_result = check_generation_result(song_id)
            try:
                if type(generation_result['data']) == list:
                    generation_result['data'] = generation_result['data'][0]
                if generation_result['data']['status'] == 'complete':
                    audio_url = generation_result['data']['audio_url']
                    if audio_url:
                        file_name = f"{outdir}/{song_id}.mp3"
                        download_file_with_retry(audio_url, file_name)
                        res.append(file_name)
                        print(f"Downloaded {file_name}")
                    else:
                        print(f"No audio URL found for song ID: {song_id}")
                    break
                else:
                    time.sleep(5)
            except Exception as e:
                raise Exception([e, traceback.format_exc(), generation_result])
    return res