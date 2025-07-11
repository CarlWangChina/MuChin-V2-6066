import sys
_rid = sys.argv[1]
_msn = int(sys.argv[2])
_tmc = int(sys.argv[3])
_cmdir = 'ali_l2p'
_lfdir = f'./log/{_cmdir}'
_dodir = f'./data/ouput/{_cmdir}'
_cmfnp = f'r{_rid}_m{_msn}-{_tmc}'
import os
os.makedirs(_lfdir, exist_ok=True)
os.makedirs(_dodir, exist_ok=True)
import logging
logging.basicConfig(
    format='%(asctime)s|%(levelname)s|%(process)d-%(thread)d|%(name)s: %(message)s',
    filename=f'{_lfdir}/{_cmfnp}.log',
    encoding='utf-8',
    level=logging.INFO)
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import TextIOWrapper as TIOW
from multiprocessing import Pool, cpu_count
import mysql.connector
from dotenv import find_dotenv, load_dotenv
from llmrelate import oai_px_skey, pxg_req_wu, spt_rl2ppt1
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
_ust_d = ""
_ust_u0 = ""
_ust_u1 = ""
_ust_u2 = ""
def _ensure_sk() -> PooledMySQLConnection:
    cnx = _cnxp.get_connection()
    cursor = cnx.cursor()
    cursor.execute(f"")
    skr = cursor.fetchone()
    if not skr: raise Exception('Should insert a new round row to table `oai_key_usage` or check the 3rd param from passed in!')
    if skr[0] != oai_px_skey: raise Exception('You must be forget to change the `OAI_PX_API_KEY` in .env!')
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
    cursor.execute(_st_qry, (_tmc, _msn,))
    songs = list[dict[str, str]]()
    for (id, lur) in cursor:
        songs.append({'id': str(id), 'lur': str(lur)})
    cursor.close()
    _logger.info(f'total {len(songs)} songs queried for machine{_msn}.')
    _ud_oku_st(cnx)
    return songs
def _try_get_cnx(sid, retry: int = 1024) -> PooledMySQLConnection | None:
    cnx = None
    tc = 0
    while cnx is None and tc <= retry:
        tc += 1
        try:
            cnx = _cnxp.get_connection()
        except:
            _logger.exception(f'SID-{sid}, GET DBCXN FAILED AT {tc}th try')
            if tc <= retry:
                _logger.warning(f'Sid-{sid}, wait 2 sec to retry')
                time.sleep(2)
    return cnx
def _wf_ladd(lock, l: list, i: int, f: TIOW, d: dict):
    f.write(json.dumps(d, ensure_ascii=False, separators=(',', ':'))+'\n')
    f.flush()
    with lock: l[i] += 1
def _core_dbw(sid, rpc, usage, sio: TIOW, eio: TIOW, stst: list, lock):
    cnx = _try_get_cnx(sid)
    if not cnx:
        _wf_ladd(lock, stst, 0, eio, {
            'id': sid, 'er': 'DBC FETCH FAILED',
            'rpc': rpc, 'usg': list(usage)
        })
        return
    errord = None
    try:
        cursor = cnx.cursor()
        udt = time.time(); p, c, t = usage
        cursor.execute(_ust_u1, (p, c, t, udt, _rid,))
        if rpc:
            cursor.execute(_ust_d, (rpc, udt, sid,))
            _wf_ladd(lock, stst, 1, sio, {'id': sid, 'rpc': rpc})
        else:
            _logger.error(f'SID: {sid}, LLM RETURNED None or Empty')
            _wf_ladd(lock, stst, 0, eio, {'id': sid, 'er': 'NoE rpc'})
        cnx.commit()
        cursor.close()
        cnx.close()
    except:
        _logger.exception(f'SID-{sid}, DB COMMIT EXCEPTION1')
        errord = {'id': sid, 'er': 'DB-CMT ERR',
                  'rpc': rpc, 'usg': list(usage)}
    if errord: _wf_ladd(lock, stst, 0, eio, errord)
def _core_job(sdict, sio: TIOW, eio: TIOW, stst: list[int], lock):
    sid = str(sdict["id"])
    fpct = "{:.3%}".format((stst[0]+stst[1]) / stst[2])
    _logger.info(f'cur-progress of sp{stst[3]}: {fpct} (E-{stst[0]}, S-{stst[1]}, T-{stst[2]}), starting job of {sid}')
    lur = str(sdict['lur']).strip()
    if not lur or len(lur) < 48:
        _logger.warning('Empty or too short lrc source, Should check column `lrc_text` of id({sid})!')
        if cnx := _try_get_cnx(sid):
            try:
                cursor = cnx.cursor()
                cursor.execute(_ust_d, ('-E0', time.time(), sid))
                cnx.commit()
                cursor.close()
                cnx.close()
            except:
                _logger.exception(f'SID-{sid}, DB COMMIT EXCEPTION0')
        _wf_ladd(lock, stst, 0, eio, {'id': sid, 'er': 'EoTS sLRC'})
        return
    rpc = None; usage = None
    try:
        rpc, usage = pxg_req_wu(spt_rl2ppt1, lur)
        _logger.info(f'sid-{sid} tu={usage}, rpc={rpc}')
    except:
        _logger.exception(f'SID-{sid}, NETWORK EXCEPTION')
        _wf_ladd(lock, stst, 0, eio, {'id': sid, 'er': 'REQ ERR'})
        return
    if usage is not None:
        _core_dbw(sid, rpc, usage, sio, eio, stst, lock)
def _mthp_start(pi: int, songs: list[dict[str, str]]):
    _logger.info(f'subprocess{pi} with: {songs[0]["id"]}..{songs[-1]["id"]}')
    sfp = f'{_dodir}/{_cmfnp}_sp{pi}.jsonl'
    efp = f'{_dodir}/{_cmfnp}_sp{pi}_err.jsonl'
    sfio = open(sfp, 'a', encoding='utf-8')
    efio = open(efp, 'a', encoding='utf-8')
    ststic = [0, 0, len(songs), pi]
    sclock = threading.Lock()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for song in songs:
            pool.submit(_core_job, song, sfio, efio, ststic, sclock)
    efio.close()
    sfio.close()
    _logger.info(f"subprocess{pi} saved {ststic[1]} results in {sfp}")
    _logger.info(f"subprocess{pi} has {ststic[0]} errors, see: {efp}")
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