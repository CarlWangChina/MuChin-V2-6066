import sys
_msn = int(sys.argv[1])
_tmc = int(sys.argv[2])
_cmfnp = f'm{_msn}-{_tmc}'
_lfdir = f'./log/inm_zhrsc/{_cmfnp}'
from datetime import datetime
_sfdtn = datetime.now().strftime('%y%m%d%H%M')
import os
os.makedirs(_lfdir, exist_ok=True)
import logging
logging.basicConfig(
    format='%(asctime)s|%(levelname)s|%(process)d-%(thread)d|%(name)s: %(message)s',
    filename=f'{_lfdir}/{_sfdtn}.log',
    encoding='utf-8',
    level=logging.INFO)
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count
import mysql.connector
from dotenv import find_dotenv, load_dotenv
from Code_for_Experiment.Targeted_Training.Lyric_Generation_LLMs.lyric_cleaning.mb_lyric.lrcleaner import MsMuscle
from mysql.connector.pooling import PooledMySQLConnection
_logger = logging.getLogger(__name__)
load_dotenv(find_dotenv(), override=True)
_dbconfig = {
    "host":     os.getenv('DB_HOST'),
    "port":     os.getenv('DB_PORT'),
    "user":     os.getenv('DB_USER'),
    "password": os.getenv('DB_PSWD'),
    "database": "mysong",
}
_cnxp = mysql.connector.pooling.MySQLConnectionPool(
    pool_name="mypool_e",
    pool_size=32,
    pool_reset_session=False,
    **_dbconfig)
_st_qry = "" 
_st_iou = "" 
def _load_songs() -> list[dict[str, str]]:
    cnx = _cnxp.get_connection()
    cursor = cnx.cursor()
    cursor.execute(_st_qry, (_tmc, _msn,))
    songs = []
    for (id, rl) in cursor:
        songs.append({'id': str(id), 'rl': str(rl)})
    cursor.close()
    _logger.info(f'total {len(songs)} songs queried for machine{_msn}.')
    cnx.close()
    return songs

def _try_get_cnx(sid, retry: int = 1024) -> PooledMySQLConnection | None:
    cnx = None
    tc = 0
    while cnx is None and tc <= retry:
        tc += 1
        try:
            cnx = _cnxp.get_connection()
        except:
            _logger.exception(f'SID-{sid}, GET DBCXN FAILED IN {tc} try')
            if tc <= retry:
                _logger.warning(f'Sid-{sid}, wait 2 sec to retry')
                time.sleep(2)
    return cnx

_msmuscle = MsMuscle()
_regft = r' *, *\d{2}:\d{2}(.\d{2,3})?'

def _core_job(sdict, stst: list[int], lock):
    sid = str(sdict["id"])
    fpct = "{:.3%}".format((stst[0]+stst[1]) / stst[2])
    _logger.info(f'cur-progress of sp{stst[3]}: {fpct} (E-{stst[0]}, '
                 f'S-{stst[1]}, T-{stst[2]}), starting job of {sid}')
    try:
        lrc = re.sub(_regft, '', sdict['rl']).strip()
        lrcstr = _msmuscle.wash_lrc_wosl(lrc)
        zhc = 0
        for c in lrcstr:
            if '\u4e00' <= c <= '\u9fff': zhc += 1
        zhr = 0.0 if not lrcstr else zhc / len(lrcstr)
        if cnx := _try_get_cnx(sid, 64):
            cts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor = cnx.cursor()
            cursor.execute(_st_iou, (sid, cts, 'NULL', zhr, zhr, cts,))
            cnx.commit()
            cursor.close()
            cnx.close()
        else:
            raise Exception('Can not insert or update to DB')
        _logger.info(f'done job for {sid}')
        with lock: stst[1] += 1
    except:
        _logger.exception(f'SID-{sid}, EXCEPTION')
        with lock: stst[0] += 1

def _mthp_start(pi: int, songs: list[dict[str, str]]):
    _logger.info(f'subprocess{pi} with: {songs[0]["id"]}..{songs[-1]["id"]}')
    ststic = [0, 0, len(songs), pi]
    sclock = threading.Lock()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for song in songs:
            pool.submit(_core_job, song, ststic, sclock)
    _logger.info(f'subprocess{pi} finished')

def _mpmt_start(cpu_c):
    _logger.info(f'start multiprocesses at pid: {os.getpid()}')
    pp = Pool(cpu_c)
    songs = _load_songs()
    epc = math.floor(len(songs)/cpu_c)
    for i in range(1, cpu_c+1):
        ss = (i-1) * epc
        cpsongs = songs[ss:] if i == cpu_c else songs[ss:i*epc]
        _logger.info(f'allocate {len(cpsongs)} songs to subprocess{i}, '
                     f'which first id is {cpsongs[0]["id"]} '
                     f'and last id is {cpsongs[-1]["id"]}')
        pp.apply_async(_mthp_start, args=(i, cpsongs,))
    pp.close()
    pp.join()
    _logger.info('all multiprocesses completed')

if __name__ == "__main__":
    cc = cpu_count()
    ucc = 1 if cc < 1 else cc
    _logger.info(f'try to use {ucc} cpu-cores on machine{_msn}.')
    _mpmt_start(ucc)
    print("Nothing bad happen for running ps, check detail log.")
    _logger.info(f'machine{_msn} has done multiprocess jobs.')