# MuChin Dataset

## About The Dataset

The MuChin dataset provides a comprehensive collection of Chinese music annotations. It features extensive descriptions from both amateur and professional perspectives, alongside detailed structural metadata such as musical sections and rhyme schemes.

We encourage scholars and researchers to utilize this resource in their research initiatives. If you use this dataset in your academic publications, we would appreciate a proper citation.

## Dataset Releases

The dataset is available in two versions:

### MuChin 1000

This set contains 1,000 audio tracks in WAV format, accompanied by detailed text annotations for:
* Professional descriptions
* Amateur descriptions
* Musical segment structure
* Rhyme patterns

Datasets-for-MuChin-V2/MuChin-1000-mp3-audio.zip
**Download Links:**
* **Hugging Face:** [karl-wang/MuChin1k](https://huggingface.co/datasets/karl-wang/MuChin1k/tree/main)
* **Baidu Netdisk:** [Download Link](https://pan.baidu.com/s/1D4xGQhYUwWbpaHyAS71dfw?pwd=1234) (Password: `1234`)



### MuChin 6066

This is the second version of the dataset, containing 6,066 (5790+1000-724) audio files. Due to its large size (several dozen gigabytes), the complete dataset is hosted on external cloud storage.

Datasets-for-MuChin-V2/MuChin-6066-audio+allAnnotations-correctRawFilev2.tar.bz2
**Download Links:**
* **Hugging Face:** [karl-wang/MuChin-v2-6066](https://huggingface.co/datasets/karl-wang/MuChin-v2-6066/resolve/main/MuChin-6066-audio%2BallAnnotations-correctRawFilev2.tar.bz2)
* **Baidu Netdisk:** [Download Link](https://pan.baidu.com/s/1GP5p3Ip_j0MraAcA52gIuQ) (Password: `68c1`)

Datasets-for-MuChin-V2/MuChin-6066-Full-Metadata-Lyrics-Sections-Rhymes-Descriptions-NoAudio.db
**Download Links:**
* **Hugging Face:** [karl-wang/MuChin-v2-6066](https://huggingface.co/datasets/karl-wang/MuChin-v2-6066/resolve/main/MuChin-6066-Full-Metadata-Lyrics-Sections-Rhymes-Descriptions-NoAudio.db)




---

## ⚠️ Important Notes

Please read these notes carefully before using the dataset.

### 1. Timestamp Discrepancy

The primary focus of our manual annotation was on descriptive and structural elements (descriptions, musical sections, rhymes), not on timestamp correction.

* **`raw_lyric`**: Contains lyric text and corresponding timestamps exported directly from KTV (Karaoke) files. These timestamps are **more accurate** but may still contain minor offset issues that were not part of our correction process. **If you require timestamps for your work, we recommend using this file.**
* **`tknz_json`**: Contains musical segment information and timestamps. The musical segment data in this file is correct. However, the **timestamps are incorrect**. This is because many musical phrases were manually merged or split during annotation, leading to a line count that no longer matches the original timestamped lyrics.

**TL;DR:** For the most reliable timestamps, use `raw_lyric`. The core value of this dataset lies in the four manually annotated aspects: colloquial descriptions, professional descriptions, musical segment labels, and rhyme labels.

### 2. Duplicate Annotations in Releases

Due to an operational error during our second round of annotation, 724 songs that were already part of the initial `MuChin 1000` release were annotated a second time.

* These 724 songs have four description files: two from the first annotation pass (professional + amateur) and two from the second.
* The total number of unique songs in the larger release is 6,066 (`5790 + 1000 - 724 = 6066`).
* The file `muchin_5790_1000_overlap.jsonl` contains the list of these 724 duplicate songs.

**Recommendation:** When using the full dataset, for these 724 songs, we recommend you **use the annotation results from the `MuChin 1000` directory and skip the duplicate results** found in the `MuChin 5790` set. We apologize for this inconvenience.

### 3. File Compression Format

The `MuChin 5790` dataset is packaged as a `.tar` archive. We chose this format over `.zip` to prevent extraction errors that occurred on some systems with large zip archives. Standard utilities on Linux, macOS, and Windows (via WSL or 7-Zip) can decompress `.tar` files without issue.

---

## Data Structure and File Descriptions

### Metadata
* **`meta_info`**: Contains song metadata, including song name, artist, album, and release year. Some fields may be missing.
* **`qidmap.json` / `subqidmap.json` / `opidmap.json`**: These files map the question, sub-question, and option IDs found in the description files to their corresponding text content.

### Music Description Annotations
* **`prof_desc_json`**: Contains structured descriptions from a professional music perspective.
* **`amat_desc_json`**: Contains structured descriptions from an amateur (colloquial) perspective.

### Musical Structure & Lyric Annotations
* **`raw_lyric`**: Provides the lyrics with the most accurate available timestamps. *(See "Timestamp Discrepancy" note above).*
* **`str_lyric`**: Contains the lyrics along with their annotated musical segment structure (e.g., intro, verse, chorus).
* **`str_rhyme`**: Indicates the positions of identified rhymes within the lyrics.
* **`tknz_json`**: Provides musical segment information for each line of lyrics. *(See "Timestamp Discrepancy" note above for timestamp warning).*

---

## Changelog / Errata

* **`raw_lyric` File Correction**: In a previous version, the `raw_lyric` file incorrectly contained descriptive information (`desc_info`) due to a file I/O error. This has been **corrected in the current version**. The `raw_lyric` file now correctly contains the timestamped lyrics as intended. Please ensure you are using the latest version of the dataset.