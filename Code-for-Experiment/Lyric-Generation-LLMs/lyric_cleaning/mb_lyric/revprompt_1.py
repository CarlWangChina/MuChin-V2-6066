import json
import logging
import os
import sys

_msn = int(sys.argv[1])
_tmc = int(sys.argv[2])
_rid = sys.argv[3]
_fmk = f'm{_msn}-{_tmc}_r{_rid}'
os.makedirs('./log/', exist_ok=True)
logging.basicConfig(
    format='%(asctime)s|%(levelname)s|%(process)d-%(thread)d|%(name)s: %(message)s',
    filename=f'./log/ali_l2p_{_fmk}.log',
    encoding='utf-8',
    level=logging.INFO)
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from multiprocessing import Pool, cpu_count
import mysql.connector
from dotenv import find_dotenv, load_dotenv
from llmrelate import gpt_req_wu, oai_askey, spt_rl2ppt
from mysql.connector.pooling import PooledMySQLConnection

_logger = logging.getLogger(__name__)
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
    **_dbconfig)
_ust_d = _ust_u0 = _ust_u1 = _ust_u2 = None

def _ensure_sk() -> PooledMySQLConnection:
    cnx = _cnxp.get_connection()
    cursor = cnx.cursor()
    cursor.execute(f"SELECT api_key FROM oai_key_usage WHERE round_id = '{_rid}'")
    skr = cursor.fetchone()
    if not skr:
        raise Exception('Should insert a new round row to table `oai_key_usage` or check the 3rd param from passed in!')
    if skr[0] != oai_askey:
        raise Exception('You must be forget to change the `OPENAI_API_KEY` in .env!')
    _logger.info(f'confirmed round id {_rid} with key {skr[0]}.')
    cursor.close()
    return cnx

def _ud_oku_st(cnx: PooledMySQLConnection):
    sts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor = cnx.cursor()
    cursor.execute(_ust_u0, (sts, _rid,))
    cnx.commit()
    cursor.close()
    cnx.close()

def _ud_oku_ct():
    cts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cnx = _cnxp.get_connection()
    cursor = cnx.cursor()
    cursor.execute(_ust_u2, (cts, _rid,))
    cnx.commit()
    cursor.close()
    cnx.close()

def _load_songs() -> list[dict[str, str]]:
    cnx = _ensure_sk()
    cursor = cnx.cursor()
    cursor.execute("SELECT id, lrc_text FROM songs WHERE round_id = %s", (_rid,))
    songs = []
    for (id, lur) in cursor:
        songs.append({'id': str(id), 'lur': str(lur)})
    cursor.close()
    _logger.info(f'total {len(songs)} songs queried for machine{_msn}.')
    _ud_oku_st(cnx)
    return songs

def _try_get_cnx(sid, retry: int = 1024) -> PooledMySQLConnection | None:
    cnx = None
    rtc = 0
    while cnx is None and rtc <= retry:
        rtc = rtc + 1
        try:
            cnx = _cnxp.get_connection()
        except:
            _logger.exception(f'SID-{sid}, GET DBCXN FAILED')
            _logger.warning(f'Sid-{sid}, wait 5 sec to retry')
            time.sleep(5)
    return cnx

def _core_dbw(sid, rpc, usage, results: list, rslock, errors: list, erlock):
    cnx = _try_get_cnx(sid)
    if cnx is None:
        with erlock:
            errors.append({
                'id': sid, 'er': 'DBC FETCH FAILED',
                'rpc': rpc, 'usg': list(usage)
            })
        return
    errord = None
    try:
        cursor = cnx.cursor()
        udt = time.time(); p, c, t = usage
        cursor.execute(_ust_u1, (p, c, t, udt, _rid,))
        if rpc is None:
            cnx.commit()
            cursor.close()
            _logger.error(f'SID: {sid}, LLM RETURNED None')
            with erlock:
                errors.append({'id': sid, 'er': 'NONE rpc'})
            return
        rpc_l = len(rpc)
        match [96 < rpc_l, rpc_l < 1024]:
            case [False, True]:
                errord = {'id': sid, 'er': 'TOO SHORT rpc', 'rpc': rpc}
                _logger.error(f'SID: {sid}, TOO SHORT rpc:\n{rpc}')
            case [True, False]:
                errord = {'id': sid, 'er': 'TOO LONG rpc', 'rpc': rpc}
                _logger.error(f'SID: {sid}, TOO LONG rpc:\n{rpc}')
            case _:
                cursor.execute(_ust_d, (rpc, udt, sid))
                with rslock:
                    results.append({'id': sid, 'rpc': rpc})
        cnx.commit()
        cursor.close()
        cnx.close()
    except:
        _logger.exception(f'SID-{sid}, DB COMMIT EXCEPTION')
        errord = {'id': sid, 'er': 'DB-CMT ERR',
                  'rpc': rpc, 'usg': list(usage)}
    if errord:
        with erlock:
            errors.append(errord)

def _core_job(sdict, results: list, rslock, errors: list, erlock):
    sid = str(sdict["id"])
    _logger.info(f'reverting song {sid}, state in set: S-{len(results)}, E-{len(errors)}')
    lur = sdict['lur']
    if not lur or len(lur) < 48:
        _logger.warning('Empty or too short lrc source, Should check column `lrc_text` of id({sid})!')
        with erlock:
            errors.append({'id': sid, 'er': 'EoTS SRC lur'})
        return
    rpc = None; usage = None
    try:
        rpc, usage = gpt_req_wu(spt_rl2ppt, lur)
    except:
        _logger.exception(f'SID-{sid}, NETWORK EXCEPTION')
        with erlock:
            errors.append({'id': sid, 'er': 'REQ ERR'})
        return
    if usage is not None:
        _core_dbw(sid, rpc, usage, results, rslock, errors, erlock)

def _mthp_start(pi: int, songs: list[dict[str, str]]):
    _logger.info(f'subprocess{pi} with: {songs[0]["id"]}..{songs[-1]["id"]}')
    rl = threading.Lock()
    el = threading.Lock()
    ra = []
    ea = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for s in songs:
            pool.submit(_core_job, s, ra, rl, ea, el)
    dir = './data/ouput/'
    os.makedirs(dir, exist_ok=True)
    sf = f'{dir}ali_l2p_{_fmk}_sp{pi}.json'
    with open(sf, 'w', encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False)
    _logger.info(f"subprocess{pi} saved {len(ra)} in result file: {sf}")
    if len(ea) > 0:
        ef = f'{dir}ali_l2p_{_fmk}_sp{pi}_errs.json'
        with open(ef, 'w', encoding="utf-8") as f:
            json.dump(ea, f, ensure_ascii=False)
        _logger.warning(f"subprocess{pi} has {len(ea)} errors, see: {ef}")
    _logger.info(f'subprocess{pi} finished')

def _mpmt_start(cpu_c):
    _logger.info(f'start multiprocesses at pid: {os.getpid()}')
    pp = Pool(cpu_c)
    songs = _load_songs()
    epc = math.floor(len(songs)/cpu_c)
    for i in range(1, cpu_c+1):
        ss = (i-1) * epc
        cpsongs = songs[ss:] if i == cpu_c else songs[ss:i*epc]
        _logger.info(f'allocate {len(cpsongs)} songs to subprocess{i}, which first id is {cpsongs[0]["id"]} and last id is {cpsongs[-1]["id"]}')
        pp.apply_async(_mthp_start, args=(i, cpsongs,))
    pp.close()
    pp.join()
    _ud_oku_ct()
    _logger.info('all multiprocesses completed')

if __name__ == "__main__":
    cc = cpu_count()
    ucc = 1 if cc < 1 else cc
    _logger.info(f'try to use {ucc} cpu-cores on machine{_msn}.')
    _mpmt_start(ucc)
    print("Nothing bad happen for running ps, check detail log.")
    _logger.info(f'machine{_msn} has done multiprocess jobs.')