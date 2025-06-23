import os
from mysql.connector import connect
from mysql.connector.pooling import PooledMySQLConnection
import logging
logging.basicConfig(format="%(asctime)s:%(name)s:[%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
config = {
    'host': 'ama-prof-divi-dev.cqydslutuswr.us-west-1.rds.amazonaws.com',
    'database': 'mysong'
}

def extract_mp3_id_map(conn: PooledMySQLConnection, output_file: str):
    query = "SELECT `id`, `mp3_id` from `ali_check_data_from_xiaoyu`"
    with conn.cursor() as cursor:
        cursor.execute(query)
        with open(output_file, 'w') as f:
            for row in cursor:
                db_id, mp3_id = row
                f.write(f"\"{mp3_id}\",{db_id}\n")

def extract_dy_qq_ids(conn: PooledMySQLConnection, output_file: str):
    query = "SELECT `song_id` from `dy_qq_all`"
    with conn.cursor() as cursor:
        cursor.execute(query)
        with open(output_file, 'w') as f:
            for row in cursor:
                song_id = row[0]
                f.write(f"\"{song_id}\"\n")

def extract_cb_ids(conn: PooledMySQLConnection, output_file: str):
    query = "SELECT `songId` from `indexed_music`"
    with conn.cursor() as cursor:
        cursor.execute(query)
        with open(output_file, 'w') as f:
            for row in cursor:
                song_id = int(row[0])
                f.write(f"{song_id}\n")

def extract_ali_cdfx_trng_lrc_ids(conn: PooledMySQLConnection, output_file: str):
    query = "SELECT `id` from `ali_cdfx_trng_lrc`"
    with conn.cursor() as cursor:
        cursor.execute(query)
        with open(output_file, 'w') as f:
            for row in cursor:
                db_id = row[0]
                f.write(f"{db_id}\n")

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
        logger.info("Extracting mp3 id map.")
        extract_mp3_id_map(db_conn, "mp3_id_map.csv")
        logger.info("Extract database ids from lyric table `ali_cdfx_trng_lrc`.")
        extract_ali_cdfx_trng_lrc_ids(db_conn, "cdfx_trng_lrc.csv")
        logger.info("Extracting QQ, Douyin ids.")
        extract_dy_qq_ids(db_conn, "dy_qq_all_ids.csv")
        logger.info("Extracting Changba ids.")
        extract_cb_ids(db_conn, "cb_ids.csv")