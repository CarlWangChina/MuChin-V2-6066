import sqlite3
from pathlib import Path
from tqdm.auto import tqdm
from ama_prof_divi_text.phoneme import PhonemeTokenizer
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger

logger = get_logger(__name__)
PACKED_DB_FILE = Path("/mnt/d/Data/Music/packed-lyrics.sqlite3")
phoneme_tokenizer = PhonemeTokenizer()

def main():
    count_statement = "SELECT COUNT(*) FROM lyrics WHERE status == 3;"
    select_statement = "SELECT song_id, phoneme FROM lyrics WHERE status == 3;"
    update_statement = "UPDATE lyrics SET tokens = ?, status = 4 WHERE song_id = ?;"
    conn = sqlite3.connect(PACKED_DB_FILE)
    cursor = conn.cursor()
    cursor_upd = conn.cursor()
    cursor.execute(count_statement)
    total_count = cursor.fetchone()[0]
    cursor.execute(select_statement)
    pbar = tqdm(total=total_count, desc="Convert Phoneme to Tokens")
    while True:
        row = cursor.fetchone()
        if row is None:
            break
        song_id = row[0]
        tokens = phoneme_tokenizer(row[1])
        tokens_str = []
        for t in tokens:
            tokens_str.append(str(t.item()))
        tokens_str = " ".join(tokens_str)
        cursor_upd.execute(update_statement, (tokens_str, song_id))
        pbar.update(1)
    conn.commit()
    cursor_upd.close()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()