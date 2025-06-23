from pydub import AudioSegment
import os

SECTION_MAPPING = {
    "主歌": "verse",
    "副歌": "chorus",
    "前奏": "intro",
    "尾奏": "outro",
    "桥段": "bridge"
}

ELEMENT_MAPPING = {
    "前奏弱起": "intro_upbeat"
}

def cut_song(file_path, cuts):
    song = AudioSegment.from_file(file_path)
    for section_name, start_time in cuts.items():
        end_time = cuts.get(section_name + "_end", None)
        if end_time is not None:
            section = song[start_time:end_time]
            section.export(f"{section_name}.mp3", format="mp3")
        else:
            print(f"No end time for section {section_name}")