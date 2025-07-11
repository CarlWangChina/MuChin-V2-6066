import json
import os
import time
from glob import glob
import mysql.connector
from dotenv import find_dotenv, load_dotenv

fs = glob('/export/efs/projects/mb-lyric/data/ouput/inm_l2p/r35_m*-4_sp1_err.jsonl', recursive=True)
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

data: list[tuple] = []
for f in fs:
    with open(f, 'r', encoding='utf-8') as fin:
        for line in fin:
            d = json.loads(line)
            t = time.time()
            data.append((d['rpc'], t, d['id'],))

with _cnxp.get_connection() as cnx:
    with cnx.cursor() as cursor:
        cursor.executemany("INSERT INTO _st_iou (rpc, t, id) VALUES (%s, %s, %s)", data)
        cnx.commit()