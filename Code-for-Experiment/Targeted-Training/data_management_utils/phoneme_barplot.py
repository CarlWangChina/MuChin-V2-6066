import sqlite3
import matplotlib.pyplot as plt
from pathlib import Path
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger

logger = get_logger(__name__)
PACKED_DB_FILE = Path("/mnt/d/Data/Music/packed-lyrics.sqlite3")

def main():
    select_statement = "SELECT phoneme, count FROM phoneme_histograms ORDER BY count DESC"
    with sqlite3.connect(PACKED_DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(select_statement)
        rows = cursor.fetchall()
        phoneme_counts = [(row[0], row[1]) for row in rows]
        total_count = sum(row[1] for row in rows)
        cursor.close()
    phonemes, counts = zip(*phoneme_counts)
    colors = ['blue' for _ in range(len(phonemes))]
    s = 0
    for i in range(len(phonemes)):
        s += counts[i]
        if s > total_count * 0.9:
            print("i= ", i)
            break
        colors[i] = 'red'
    plt.figure(figsize=(10, 6))
    plt.bar(phonemes, counts, color=colors)
    plt.title('Phoneme Histogram')
    plt.xlabel('Phoneme')
    plt.ylabel('Count')
    plt.show()

if __name__ == "__main__":
    main()