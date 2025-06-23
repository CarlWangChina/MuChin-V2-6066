import opencc
import re
import sqlite3
import string
from typing import Optional
from pathlib import Path
from tqdm.auto import tqdm
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger

logger = get_logger(__name__)
PACKED_DB_FILE = Path("/mnt/d/Data/Music/packed-lyrics.sqlite3")
opencc_converter = opencc.OpenCC("t2s")
matcher = re.compile(r"Text:(.*?) Timestamps:\((.*), (.*)\)")
filter_pattern = re.compile(r"^[0-9\s\-{}]*$".format(re.escape(string.punctuation)))
hallucinations = []
hallucination_table_path = Path(__file__).parent.joinpath('hallucinations.txt')
with open(hallucination_table_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line == "" or line.startswith(""):
            continue
        hallucinations.append(line)

def _filter_lyrics_text(text: str) -> str:
    text = re.sub(r'\(.*?\)', '', text)
    for hallucination in hallucinations:
        if hallucination.lower() in text:
            return ""
    if filter_pattern.match(text):
        return ""
    text = text.strip()
    if len(text) == 1 or len(text) >= 80:
        return ""
    return text

def _clean_lyrics(lyrics: str) -> Optional[str]:
    lines = lyrics.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line == "":
            continue
        match = matcher.match(line)
        if match is None:
            logger.warning("Bad lyrics line: %s", line)
            continue
        text = match.group(1).strip()
        text = opencc_converter.convert(text).lower()
        text = _filter_lyrics_text(text)
        if text == "":
            continue
        cleaned_lines.append(text)
    result = " ".join(cleaned_lines)
    if result == "":
        return None
    return result

def main():
    count_statement = "SELECT COUNT(*) FROM lyrics WHERE status == 1;"
    select_statement = "SELECT song_id, lyrics FROM lyrics WHERE status == 1;"
    update_statement = "UPDATE lyrics SET cleaned_up_lyrics = ?, status = 2 WHERE song_id = ?;"
    conn = sqlite3.connect(PACKED_DB_FILE)
    cursor = conn.cursor()
    cursor_upd = conn.cursor()
    cursor.execute(count_statement)
    total_count = cursor.fetchone()[0]
    cursor.execute(select_statement)
    pbar = tqdm(total=total_count, desc="Cleaning up lyrics")
    while True:
        row = cursor.fetchone()
        if row is None:
            break
        song_id = row[0]
        lyrics = row[1]
        cleaned_lyrics = _clean_lyrics(lyrics)
        if cleaned_lyrics is None:
            pbar.update(1)
            continue
        cursor_upd.execute(update_statement, (cleaned_lyrics, song_id))
        conn.commit()
        pbar.update(1)
    pbar.close()
    cursor.close()
    cursor_upd.close()
    conn.close()

if __name__ == "__main__":
    main()