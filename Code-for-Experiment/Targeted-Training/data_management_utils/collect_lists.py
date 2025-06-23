import csv
import sqlite3
from pathlib import Path
from tqdm.auto import tqdm
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger

logger = get_logger(__name__)
NUM_NODES = 8
LIST_TYPES = ["wav", "vae", "clap", "lyric"]
LISTS_DIR = Path("/mnt/d/Data/Music/lists")
DB_FILE = Path("/mnt/d/Data/Music/packed-lists.sqlite3")

def main():
    assert LISTS_DIR.exists(), f"Directory {LISTS_DIR} does not exist"
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    song_list = {}
    for i in range(NUM_NODES):
        for list_type in LIST_TYPES:
            list_file = LISTS_DIR / f"{list_type}_list_{i}.csv"
            assert list_file.exists(), f"File {list_file} does not exist"
            with open(list_file, "r") as csv_file:
                reader = csv.reader(csv_file)
                next(reader)
                rows = list(reader)
                for row in tqdm(rows, desc=f"Processing {list_type} on node {i}"):
                    dataset = row[0]
                    song_id = row[1]
                    if song_id not in song_list:
                        song_list[song_id] = {
                            "song_id": song_id,
                            "dataset": dataset,
                            "node": i,
                            "has_wav": False,
                            "has_vae": False,
                            "has_clap": False,
                            "has_lyric": False,
                            "wav_offset": None,
                            "wav_length": None,
                            "vae_offset": None,
                            "vae_length": None,
                            "clap_offset": None,
                            "clap_length": None,
                            "lyric_offset": None,
                            "lyric_length": None
                        }
                    song_node = song_list[song_id]
                    assert song_node["dataset"] == dataset, f"Dataset mismatch for song {song_id}"
                    assert song_node["node"] == i, f"Node mismatch for song {song_id}"
                    song_node[f"has_{list_type}"] = True
                    song_node[f"{list_type}_offset"] = int(row[2])
                    song_node[f"{list_type}_length"] = int(row[4])
    create_statement = "CREATE TABLE IF NOT EXISTS songs (song_id TEXT PRIMARY KEY, dataset TEXT, node INTEGER, has_wav INTEGER, has_vae INTEGER, has_clap INTEGER, has_lyric INTEGER, wav_offset INTEGER, wav_length INTEGER, vae_offset INTEGER, vae_length INTEGER, clap_offset INTEGER, clap_length INTEGER, lyric_offset INTEGER, lyric_length INTEGER)"
    cursor.execute(create_statement)
    conn.commit()
    create_index_statement = "CREATE INDEX IF NOT EXISTS song_id_index ON songs (song_id)"
    cursor.execute(create_index_statement)
    create_index_statement = "CREATE INDEX IF NOT EXISTS dataset_index ON songs (dataset)"
    cursor.execute(create_index_statement)
    create_index_statement = "CREATE INDEX IF NOT EXISTS node_index ON songs (node)"
    cursor.execute(create_index_statement)
    create_index_statement = "CREATE INDEX IF NOT EXISTS has_wav_index ON songs (has_wav)"
    cursor.execute(create_index_statement)
    create_index_statement = "CREATE INDEX IF NOT EXISTS has_vae_index ON songs (has_vae)"
    cursor.execute(create_index_statement)
    create_index_statement = "CREATE INDEX IF NOT EXISTS has_clap_index ON songs (has_clap)"
    cursor.execute(create_index_statement)
    create_index_statement = "CREATE INDEX IF NOT EXISTS has_lyric_index ON songs (has_lyric)"
    cursor.execute(create_index_statement)
    conn.commit()
    insert_statement = "INSERT OR IGNORE INTO songs (song_id, dataset, node, has_wav, has_vae, has_clap, has_lyric, wav_offset, wav_length, vae_offset, vae_length, clap_offset, clap_length, lyric_offset, lyric_length) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    logger.info("Writing into the database...")
    for _, s in song_list.items():
        cursor.execute(insert_statement, (
            s["song_id"],
            s["dataset"],
            s["node"],
            s["has_wav"],
            s["has_vae"],
            s["has_clap"],
            s["has_lyric"],
            s["wav_offset"],
            s["wav_length"],
            s["vae_offset"],
            s["vae_length"],
            s["clap_offset"],
            s["clap_length"],
            s["lyric_offset"],
            s["lyric_length"]
        ))
    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()