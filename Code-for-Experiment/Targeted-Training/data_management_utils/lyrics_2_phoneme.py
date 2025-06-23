import sqlite3
from typing import Optional
from pathlib import Path
from tqdm.auto import tqdm
from ama_prof_divi_text.phoneme import Lyrics2Phoneme
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger

logger = get_logger(__name__)
PACKED_DB_FILE = Path("/mnt/d/Data/Music/packed-lyrics.sqlite3")
lyrics_to_phoneme = Lyrics2Phoneme()

def _convert_lyrics_to_phoneme(lyrics: str) -> Optional[str]:
    phoneme = lyrics_to_phoneme.translate(lyrics)
    return phoneme

def calc_phoneme_histograms(conn: sqlite3.Connection):
    create_statement = """CREATE TABLE IF NOT EXISTS phoneme_histograms (
                            phoneme TEXT PRIMARY KEY,
                            count INTEGER
                        );"""
    cursor = conn.cursor()
    cursor.execute(create_statement)
    conn.commit()
    create_index_statement = """CREATE INDEX IF NOT EXISTS idx_phoneme_histograms ON phoneme_histograms (phoneme);"""
    cursor.execute(create_index_statement)
    conn.commit()
    count_statement = "SELECT COUNT(*) FROM lyrics WHERE status == 3;"
    select_statement = "SELECT phoneme FROM lyrics WHERE status == 3;"
    update_statement = "INSERT OR REPLACE INTO phoneme_histograms (phoneme, count) VALUES (?, ?);"
    cursor.execute(count_statement)
    total_count = cursor.fetchone()[0]
    phoneme_histogram = {}
    cursor.execute(select_statement)
    pbar = tqdm(total=total_count, desc="Calculating phoneme histogram")
    while True:
        row = cursor.fetchone()
        if row is None:
            break
        phoneme = row[0].split(" ")
        for p in phoneme:
            p = p.strip()
            if p != "":
                if p not in phoneme_histogram:
                    phoneme_histogram[p] = 1
                else:
                    phoneme_histogram[p] += 1
        pbar.update(1)
    for p, c in phoneme_histogram.items():
        cursor.execute(update_statement, (p, c))
    conn.commit()
    pbar.close()
    cursor.close()

def main():
    count_statement = "SELECT COUNT(*) FROM lyrics WHERE status == 2;"
    select_statement = "SELECT song_id, cleaned_up_lyrics FROM lyrics WHERE status == 2;"
    update_statement = "UPDATE lyrics SET phoneme = ?, status = 3 WHERE song_id = ?;"
    conn = sqlite3.connect(PACKED_DB_FILE)
    cursor = conn.cursor()
    cursor_upd = conn.cursor()
    cursor.execute(count_statement)
    total_count = cursor.fetchone()[0]
    cursor.execute(select_statement)
    pbar = tqdm(total=total_count, desc="Convert Lyrics to Phoneme")
    while True:
        row = cursor.fetchone()
        if row is None:
            break
        song_id = row[0]
        lyrics = row[1]
        assert lyrics is not None
        phoneme = _convert_lyrics_to_phoneme(lyrics)
        if phoneme is None:
            pbar.update(1)
            continue
        cursor_upd.execute(update_statement, (phoneme, song_id))
        conn.commit()
        pbar.update(1)
    pbar.close()
    cursor_upd.close()
    cursor.close()
    calc_phoneme_histograms(conn)
    conn.close()

if __name__ == "__main__":
    main()