import os
import csv
import pymongo
from tqdm.auto import tqdm
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

MONGO_URI = os.getenv('MONGO_URI')
MONGO_USER = os.getenv('MONGO_USER')
MONGO_PASSWORD = os.getenv('MONGO_PASSWORD')

assert MONGO_URI, "Environment variable MONGO_URI is not set."
assert MONGO_USER, "Environment variable MONGO_USER is not set."
assert MONGO_PASSWORD, "Environment variable MONGO_PASSWORD is not set."

client = pymongo.MongoClient(MONGO_URI, username=MONGO_USER, password=MONGO_PASSWORD)
db = client['training_data']

INPUT_FILE = r"/mnt/d/Data/Music/csv/240523-6.csv"
OUTPUT_FILE = r"/tmp/240523-6-duplicate.csv"
BATCH_SIZE = 100
NUM_THREADS = 8

def check_title_artists_batch(batch_req_ids: List[str], batch_titles: List[str], batch_artists: List[str]) -> Dict[str, Tuple[str, str, str, str]]:
    query = {
        'title': {'$in': batch_titles},
        'artist': {'$in': batch_artists}
    }
    result = db['data'].find(query, {'title': 1, 'artist': 1, 'tag': 1, '_id': 1})
    result_list = list(result)
    output_dict = {}
    for req_id, title, artist in zip(batch_req_ids, batch_titles, batch_artists):
        dups = ""
        for item in result_list:
            if item['title'] == title and item['artist'] == artist:
                if len(dups) > 0:
                    dups += ", "
                dups += f"[{str(item['_id'])}, {item['tag']}]"
                break
        if len(dups) > 0:
            output_dict[req_id] = (req_id, title, artist, dups)
    return output_dict

csv_ids = []
titles = []
artists = []
with open(INPUT_FILE, "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    next(reader, None)
    for row in reader:
        csv_ids.append(row[1])
        titles.append(row[3])
        artists.append(row[4])

print(f"Total number of titles: {len(titles)}")

csv_id_batches = [csv_ids[i:i + BATCH_SIZE] for i in range(0, len(csv_ids), BATCH_SIZE)]
title_batches = [titles[i:i + BATCH_SIZE] for i in range(0, len(titles), BATCH_SIZE)]
artist_batches = [artists[i:i + BATCH_SIZE] for i in range(0, len(artists), BATCH_SIZE)]

duplicates = {}
with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    futures = [executor.submit(check_title_artists_batch, csv_id_batch, title_batch, artist_batch) for csv_id_batch, title_batch, artist_batch in zip(csv_id_batches, title_batches, artist_batches)]
    with tqdm(total=len(title_batches), desc="Processing batches") as pbar:
        for future in as_completed(futures):
            duplicates.update(future.result())
            pbar.update(1)

print(f"Total number of duplicates: {len(duplicates)}")

with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["id", "title", "artist", "duplicates"])
    for key, value in duplicates.items():
        writer.writerow([value[0], value[1], value[2], value[3]])