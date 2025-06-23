import torch
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple, List, Any
from music_dit.utils import get_logger, get_audio_utils

logger = get_logger(__name__)
audio_utils = get_audio_utils()

def _walk_through_directory(path: Path, suffix: str, approx_file_nums: int, desc: str) -> Tuple[Dict[str, Any], int]:
    path_dict = {}
    num_files = 0
    pbar = tqdm(total=approx_file_nums, desc=desc)
    for file in path.rglob(f'*{suffix}'):
        num_files += 1
        sub_dirs = []
        t = file.parent
        while t != path:
            sub_dirs = [t.name] + sub_dirs
            t = t.parent
        current_dict = path_dict
        for sub_dir in sub_dirs:
            if sub_dir not in current_dict:
                current_dict[sub_dir] = {}
            current_dict = current_dict[sub_dir]
        current_dict[file.stem] = "/".join(sub_dirs) + "/" + file.name
        pbar.update(1)
    return path_dict, num_files

def _find_paired_files(audio_dict: Dict[str, Any], lyric_dict: Dict[str, Any], paired: List[Tuple[str, str]]):
    for key in audio_dict:
        audio_item = audio_dict[key]
        if key in lyric_dict:
            lyric_item = lyric_dict[key]
            if isinstance(audio_item, dict) and isinstance(lyric_item, dict):
                _find_paired_files(audio_item, lyric_item, paired)
            elif isinstance(audio_item, str) and isinstance(lyric_item, str):
                paired.append((audio_item, lyric_item))

def make_data_index(audio_dir: str, lyrics_dir: str, output_dir: str, audio_suffix: str = '.mp3', lyrics_suffix: str = '.txt', approx_audio_file_num: int = 100000, approx_lyrics_file_num: int = 100000):
    logger.info("Making data index...")
    audio_dir = Path(audio_dir)
    lyrics_dir = Path(lyrics_dir)
    output_dir = Path(output_dir)
    assert audio_dir.is_dir(), f"Audio directory '{audio_dir}' does not exist."
    assert lyrics_dir.is_dir(), f"Lyrics directory '{lyrics_dir}' does not exist."
    assert output_dir.is_dir(), f"Output directory '{output_dir}' does not exist."
    output_index_file = output_dir / 'paired-data-index.csv'
    logger.info("Walking through audio directory...")
    source_audio_files, n_files = _walk_through_directory(audio_dir, audio_suffix, approx_audio_file_num, desc="Audio files")
    logger.info("%d audio files found.", n_files)
    logger.info("Working through lyrics directory...")
    source_lyrics_files, n_files = _walk_through_directory(lyrics_dir, lyrics_suffix, approx_lyrics_file_num, desc="Lyrics files")
    logger.info("%d lyrics files found.", n_files)
    paired_files = []
    _find_paired_files(source_audio_files, source_lyrics_files, paired_files)
    logger.info("Found %d paired audio and lyric files.", len(paired_files))
    with open(output_index_file, 'w', encoding='utf-8') as f:
        for audio_file, lyric_file in tqdm(paired_files):
            try:
                f.write('"{}", "{}"\n'.format(audio_file, lyric_file))
            except Exception as e:
                logger.error("Error loading audio file '%s': %s", audio_file, e)
                continue
    logger.info("Data index files saved to '%s'.", output_dir)