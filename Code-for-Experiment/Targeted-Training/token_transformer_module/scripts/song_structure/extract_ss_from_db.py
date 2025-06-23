import os
import re
import json
from tqdm import tqdm
from pathlib import Path
from mysql.connector import connect
from mysql.connector.pooling import PooledMySQLConnection
import logging
logging.basicConfig(format="%(asctime)s:%(name)s:[%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
config = {
    'host': 'ama-prof-divi-dev.cqydslutuswr.us-west-1.rds.amazonaws.com',
    'database': 'mysong'
}

ALI_CONTROL_FILES = [
    "music-type-checked.txt",
    "music-type-checked-1087.txt"
]

CB_CONTROL_FILES = [
    "music-type-checked-cb-unique.txt",
]

DY_QQ_CONTROL_FILES = [
    "music-type-checked-dyqy.txt"
]

OUTPUT_FILE = "ss_info.json"
BAD_ID_LIST_FILE = "bad_ids.txt"

def _format_ss_lyrics(lyrics_json: dict, *, db_id: str) -> [dict]:
    pattern_tm = re.compile(r"^\[(\d+):(\d+(\.\d+)?)]$")
    pattern_tm2 = re.compile(r"^\[(\d+):(\d+(:\d+)?)]$")
    pattern_lyric = re.compile(r"^(<\|.*\|>)(.*)$")
    output = []
    for key in lyrics_json.keys():
        match = pattern_tm.match(key)
        if match is None:
            match = pattern_tm2.match(key)
        assert match is not None, f"ID={db_id}: Invalid timestamp format: {key}"
        offset = int(match.group(1)) * 60 + float(match.group(2).replace(":", "."))
        match = pattern_lyric.match(lyrics_json[key])
        assert match is not None, f"ID={db_id}: Invalid lyric (with SS) format: {lyrics_json[key]}"
        ss = match.group(1)
        lyric = match.group(2)
        output.append({
            "offset": offset,
            "ss": ss,
            "lyric": lyric
        })
    return output

def _get_id_set_from_control_files(control_file_list: [str]) -> set:
    mp3_id_set = set()
    for control_file in control_file_list:
        with open(control_file, 'r', encoding='utf-8') as fd:
            for line in fd:
                json_data = json.loads(line)
                mp3_id = Path(json_data['path']).stem
                mp3_id_set.add(mp3_id)
    return mp3_id_set

def _write_lyrics_json(mp3_id_set: set, mp3_id_map: dict, mp3_name_map: dict, query: str, conn: PooledMySQLConnection, output_fd, bad_id_fd, *, is_last_list=False):
    for i, mp3_id in enumerate(tqdm(sorted(mp3_id_set))):
        comma = "," if (not is_last_list) or (i < len(mp3_id_set) - 1) else ""
        db_id = mp3_id_map[mp3_id]
        output_json = {"song_name": mp3_name_map[mp3_id], "ss": None}
        with conn.cursor() as cursor:
            cursor.execute(query, (db_id,))
            row = cursor.fetchone()
            if row is not None and row[0] is not None and row[0] != b'':
                lyrics_json = json.loads(row[0])
                if lyrics_json != {}:
                    ss = _format_ss_lyrics(lyrics_json, db_id=db_id)
                    output_json["ss"] = ss
                else:
                    bad_id_fd.write(f"{mp3_id}\n")
                    bad_id_fd.flush()
            output_fd.write(f"    \"{mp3_id}\": {json.dumps(output_json, ensure_ascii=False)}{comma}\n")
            output_fd.flush()
            for _ in cursor:
                logger.warning(f"ID={db_id}: More than one row returned from the database.")
            cursor.close()

def extract_ss_info_ali(control_file_list: [str], conn: PooledMySQLConnection, output_fd, bad_id_fd, *, is_last_list=False):
    mp3_id_set = _get_id_set_from_control_files(control_file_list)
    mp3_id_map = {}
    mp3_name_map = {}
    query = "SELECT `id`, `mp3_id`, `music_name` from `ali_check_data_from_xiaoyu`"
    with conn.cursor() as cursor:
        cursor.execute(query)
        for row in cursor:
            db_id, mp3_id, song_name = row
            mp3_id_map[mp3_id] = db_id
            mp3_name_map[mp3_id] = song_name
    for song_id in mp3_id_set:
        assert song_id in mp3_id_map, f"MP3 ID {song_id} not found in the database."
    query = "SELECT `tkn_wi_tt_json` from `ali_cdfx_trng_lrc` where `id`=%s;"
    _write_lyrics_json(mp3_id_set, mp3_id_map, mp3_name_map, query, conn, output_fd, bad_id_fd, is_last_list=is_last_list)

def extract_ss_info_cb(control_file_list: [str], conn: PooledMySQLConnection, output_fd, bad_id_fd, *, is_last_list=False):
    mp3_id_set = _get_id_set_from_control_files(control_file_list)
    mp3_id_map = {}
    mp3_name_map = {}
    query = "SELECT `id`, `songId`, `songName` from `indexed_music`"
    with conn.cursor() as cursor:
        cursor.execute(query)
        for row in cursor:
            db_id, mp3_id, song_name = row
            mp3_id_map[str(mp3_id)] = db_id
            mp3_name_map[str(mp3_id)] = song_name
    for song_id in mp3_id_set:
        assert song_id in mp3_id_map, f"Song ID {song_id} not found in the database."
    query = "SELECT `tkn_wi_tt_json` from `inm_trng_ppd` where `id`=%s;"
    _write_lyrics_json(mp3_id_set, mp3_id_map, mp3_name_map, query, conn, output_fd, bad_id_fd, is_last_list=is_last_list)

def extract_ss_info_dyqy(control_file_list: [str], conn: PooledMySQLConnection, output_fd, bad_id_fd, *, is_last_list=False):
    mp3_id_set = _get_id_set_from_control_files(control_file_list)
    mp3_name_map = {}
    mp3_id_map = {}
    query = "SELECT `song_id`, `song_name` from `dy_qq_all`"
    with conn.cursor() as cursor:
        cursor.execute(query)
        for row in cursor:
            mp3_id, song_name = row
            mp3_id_map[mp3_id] = mp3_id
            mp3_name_map[mp3_id] = song_name
    for song_id in mp3_id_set:
        assert song_id in mp3_name_map, f"Song ID {song_id} not found in the database."
    query = "SELECT `tkn_wi_tt_json` from `dyq_trng_ppd` where `sid`=%s;"
    _write_lyrics_json(mp3_id_set, mp3_id_map, mp3_name_map, query, conn, output_fd, bad_id_fd, is_last_list=is_last_list)

if __name__ == '__main__':
    if "MYSQL_USER" in os.environ:
        config["user"] = os.environ["MYSQL_USER"]
    else:
        raise ValueError("MYSQL_USER environment variable is not set.")
    if "MYSQL_PASSWORD" in os.environ:
        config["password"] = os.environ["MYSQL_PASSWORD"]
    else:
        raise ValueError("MYSQL_PASSWORD environment variable is not set.")
    with connect(**config) as db_conn:
        logger.info("Extracting song structure to file %s.", OUTPUT_FILE)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            with open(BAD_ID_LIST_FILE, 'w', encoding='utf-8') as bad_id_f:
                f.write("{\n")
                logger.info("Extracting song structure information from the Alibaba dataset.")
                extract_ss_info_ali(ALI_CONTROL_FILES, db_conn, f, bad_id_f)
                logger.info("Extracting song structure information from the Changba dataset.")
                extract_ss_info_cb(CB_CONTROL_FILES, db_conn, f, bad_id_f)
                logger.info("Extracting song structure information from the Douyin/QQ Music dataset.")
                extract_ss_info_dyqy(DY_QQ_CONTROL_FILES, db_conn, f, bad_id_f, is_last_list=True)
                f.write("}\n")