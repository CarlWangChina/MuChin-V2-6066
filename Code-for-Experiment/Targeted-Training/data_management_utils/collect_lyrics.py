import csv
import os
import sqlite3
import urllib.parse
from pathlib import Path
from tqdm.auto import tqdm
from typing import Optional, Dict
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger

logger = get_logger(__name__)
WAV_LIST_PATH = Path("/mnt/d/Data/Music/wav-lists")
PACKED_DB_FILE = Path("/mnt/d/Data/Music/packed-lyrics.sqlite3")
LYRICS_PATHS = {
    "ali": Path("/mnt/d/Data/Music/ali-40w-vocal_datasets_txt_files_10000"),
    "cb": Path("/mnt/d/Data/Music/cb_datasets_txt_files"),
    "dyqy": Path("/mnt/d/Data/Music/dyqy-vocal_datasets_txt_files"),
    "mysong": Path("/mnt/d/Data/Music/mysong_vocal_datasets_txt_files_10000"),
}


def _index_lyric_files() -> Dict[str, Dict[str, Path]]:
    lyrics_index = {}
    for dataset, lyrics_dir in LYRICS_PATHS.items():
        logger.info("Indexing lyric files in %s...", dataset)
        if not lyrics_dir.is_dir():
            logger.error("Directory %s is not exist.", lyrics_dir)
            exit(1)
        file_count = 0
        file_index = {}
        for lyrics_file in lyrics_dir.glob("**/*.txt"):
            file_count += 1
            song_id = lyrics_file.stem
            if song_id.endswith("_src"):
                song_id = song_id[:-len("_src")]
            song_id = urllib.parse.quote(song_id)
            file_index[song_id] = lyrics_file.absolute()
        logger.info("Dataset %s indexed.  %d files found.", dataset, file_count)
        lyrics_index[dataset] = file_index
    return lyrics_index


def _get_lyrics(lyrics_index: Dict[str, Dict[str, Path]],
                dataset: str,
                song_id: str) -> Optional[str]:
    if dataset == 'ali2':
        dataset = 'ali'
    assert dataset in lyrics_index, f"Dataset {dataset} is not indexed."
    original_song_id = song_id
    song_id = Path(song_id).stem
    if song_id.endswith("_src"):
        song_id = song_id[:-len("_src")]
    if song_id not in lyrics_index[dataset]:
        logger.error(f"Song {original_song_id} is not indexed in dataset {dataset}.")
        return None
    lyrics_file = lyrics_index[dataset][song_id]
    with open(lyrics_file, "r") as fp:
        lyrics = fp.read().strip()
    return lyrics


def main():
    lyrics_index = _index_lyric_files()
    processed_count = 0
    error_count = 0
    create_table_sql = ""
    create_index_sql = ""
    insert_sql = ""
    if PACKED_DB_FILE.exists():
        logger.warning("%s already exists.  It will be overwritten.", PACKED_DB_FILE)
        os.remove(PACKED_DB_FILE)
    conn = sqlite3.connect(PACKED_DB_FILE)
    cursor = conn.cursor()
    cursor.execute(create_table_sql)
    cursor.execute(create_index_sql)
    conn.commit()
    for csv_file in WAV_LIST_PATH.glob("*.csv"):
        with open(csv_file, "r") as csv_fp:
            csv_reader = csv.reader(csv_fp)
            next(csv_reader)
            csv_list = list(csv_reader)
            for row in tqdm(csv_list, desc=f"Processing {csv_file}"):
                dataset = row[0]
                song_id = row[1]
                lyrics = _get_lyrics(lyrics_index, dataset, song_id)
                if lyrics is None:
                    cursor.execute(insert_sql, (
                        song_id,
                        dataset,
                        0,
                        None,
                        None,
                        None,
                        None,
                    ))
                    error_count += 1
                    continue
                cursor.execute(insert_sql, (
                    song_id,
                    dataset,
                    1,
                    lyrics,
                    None,
                    None,
                    None,
                ))
                processed_count += 1
        conn.commit()
    cursor.close()
    conn.close()
    logger.info("Done.  %d processed, %d errors.", processed_count, error_count)


if __name__ == "__main__":
    main()