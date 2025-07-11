import csv
import json
import os
import mysql.connector
from dotenv import find_dotenv, load_dotenv

os.makedirs('./log/ali_l2s_rc', exist_ok=True)
os.makedirs('./log/inm_l2s_rc', exist_ok=True)
os.makedirs('./data/ouput/ali_l2s_rc', exist_ok=True)
os.makedirs('./data/ouput/inm_l2s_rc', exist_ok=True)

load_dotenv(find_dotenv(), override=True)

_dbconfig = {
    "host": os.getenv('DB_HOST'),
    "port": os.getenv('DB_PORT'),
    "user": os.getenv('DB_USER'),
    "password": os.getenv('DB_PSWD'),
    "database": "mysong",
}

_cnxp = mysql.connector.pooling.MySQLConnectionPool(
    pool_name="mypool_e",
    pool_size=32,
    pool_reset_session=False,
    **_dbconfig
)

ali_mids = set()
with open('/export/efs/projects/mb-lyric/temp/ali_bad_ids.txt', 'r', encoding='utf-8') as f:
    ali_mids.update(f.read().splitlines())

inm_mids = set()
with open('/export/efs/projects/mb-lyric/temp/inm_bad_ids.txt', 'r', encoding='utf-8') as f:
    inm_mids.update(f.read().splitlines())

cnx = _cnxp.get_connection()
cursor = cnx.cursor()
cursor.execute(
    "SELECT id, mid, lt, hc, ls, ms, dis, grp, gal FROM ali_l2s_rc WHERE mid NOT IN %s",
    (tuple(ali_mids),)
)

ali_reasons = []
ali_songs = []
for (id, mid, lt, hc, ls, ms, dis, grp, gal) in cursor.fetchall():
    csr = cnx.cursor()
    csr.execute(
        "SELECT first FROM ali_check_lrc_hash WHERE id = %s",
        (id,)
    )
    first_qr = csr.fetchone()
    csr.close()
    ali_reasons.append({
        'id': id,
        'mp3_id': mid,
        'first_in_tb_lrc_hash': first_qr is not None and first_qr[0] == 1,
        'lrc_text': len(str(lt).strip()) >= 48,
        'have_chinese': hc == 1,
        'lrc_status': ls,
        'mp3_status': ms,
        'data_in_server': isinstance(dis, str) and dis != 'null',
        'gpt_rev_prompt': isinstance(grp, str) and not grp.startswith('-E'),
        'gpt_amss_lyric': isinstance(gal, str) and not gal.startswith('-E'),
    })
    ali_songs.append({'id': id, 'lur': lt})

cursor.close()

if ali_reasons:
    with open('./data/ouput/ali_l2s_rc/why_miss.csv', 'w', newline='') as f:
        csv_keys = ali_reasons[0].keys()
        dict_writer = csv.DictWriter(f, csv_keys)
        dict_writer.writeheader()
        dict_writer.writerows(ali_reasons)

with open('./data/ouput/ali_l2s_rc/songs.json', 'w', encoding='utf-8') as f:
    json.dump(ali_songs, f, ensure_ascii=False, indent=None, separators=(',', ':'))

cursor = cnx.cursor()
cursor.execute(
    "SELECT id, sid, lwt, hc FROM inm_l2s_rc WHERE sid NOT IN %s",
    (tuple(inm_mids),)
)

inm_reasons = []
inm_songs = []
for (id, sid, lwt, hc) in cursor.fetchall():
    csr = cnx.cursor()
    csr.execute(
        "SELECT lrp FROM inm_check_lrc_hash WHERE id = %s",
        (id,)
    )
    row = csr.fetchone()
    csr.close()
    reason = {
        'id': id,
        'songId': sid,
        'lyric_with_time': len(str(lwt).strip()) >= 48,
        'have_chinese': hc == 1,
    }
    if row is None:
        inm_songs.append({'id': id, 'lur': lwt})
        reason['llm_rev_prompt'] = False
    else:
        lrp = str(row[0])
        reason['llm_rev_prompt'] = not lrp.startswith('-E')
    inm_reasons.append(reason)

cursor.close()

if inm_reasons:
    with open('./data/ouput/inm_l2s_rc/why_miss.csv', 'w', newline='') as f:
        csv_keys = inm_reasons[0].keys()
        dict_writer = csv.DictWriter(f, csv_keys)
        dict_writer.writeheader()
        dict_writer.writerows(inm_reasons)

with open('./data/ouput/inm_l2s_rc/songs.json', 'w', encoding='utf-8') as f:
    json.dump(inm_songs, f, ensure_ascii=False, indent=None, separators=(',', ':'))

cnx.close()