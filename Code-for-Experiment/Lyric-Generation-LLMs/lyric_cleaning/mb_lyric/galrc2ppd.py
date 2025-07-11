import sys
msn = int(sys.argv[1])
tmc = int(sys.argv[2])
from datetime import datetime
sfdtn = datetime.now().strftime('%y%m%d%H%M')
codir = f'ali_c2ppd/{sfdtn}'
lfdir = f'./log/{codir}'
dodir = f'./data/ouput/{codir}'
import os
os.makedirs(lfdir, exist_ok=True)
os.makedirs(dodir, exist_ok=True)
import logging
logging.basicConfig(
    format='%(asctime)s|%(levelname)s|%(process)d-%(thread)d|%(name)s: %(message)s',
    filename=f'{lfdir}/m{msn}-{tmc}.log',
    encoding='utf-8',
    level=logging.INFO)
import json
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count
import mysql.connector
from dotenv import find_dotenv, load_dotenv
from mysql.connector.pooling import PooledMySQLConnection
logger = logging.getLogger(__name__)
load_dotenv(find_dotenv(), override=True)
dbconfig = {
    "host":     os.getenv('DB_HOST'),
    "port":     os.getenv('DB_PORT'),
    "user":     os.getenv('DB_USER'),
    "password": os.getenv('DB_PSWD'),
    "database": "mysong",
}
cnxp = mysql.connector.pooling.MySQLConnectionPool(
    pool_name="mypool_e",
    pool_size=32,
    pool_reset_session=False,
    **dbconfig)
_st_iou = """UPDATE song SET last_update=%s, lrc4brotao=%s, lrc4carl=%s, lrc4brotao_new=%s, lrc4carl_new=%s, last_update_new=%s WHERE id=%s"""
def _load_songs() -> list[dict[str, str]]:
    cnx = cnxp.get_connection()
    cursor = cnx.cursor()
    cursor.execute("SELECT id, raw_lrc, game_lrc FROM song")
    songs = []
    for (id, rl, gl) in cursor:
        songs.append({'id': str(id), 'rl': str(rl), 'gl': str(gl)})
    cursor.close()
    logger.info(f'total {len(songs)} songs queried for machine{msn}.')
    cnx.close()
    return songs
def _try_get_cnx(sid, retry: int = 1024) -> PooledMySQLConnection | None:
    cnx = None
    tc = 0
    while cnx is None and tc <= retry:
        tc += 1
        try:
            cnx = cnxp.get_connection()
        except:
            logger.exception(f'SID-{sid}, GET DBCXN FAILED IN {tc} try')
            if tc <= retry:
                logger.warning(f'Sid-{sid}, wait 2 sec to retry')
                time.sleep(2)
    return cnx
others_rg = r'(?i)^others\s*\|\s*'
def _group_oindices(glines: list[str]) -> list[set[int]]:
    result = []
    cgroup = []
    for i, line in enumerate(glines):
        if re.match(others_rg, line.lstrip()):
            if not cgroup or i == cgroup[-1] + 1:
                cgroup.append(i)
            else:
                result.append(set(cgroup))
                cgroup = [i]
    if cgroup:
        result.append(set(cgroup))
    return result
def _tsf_mss4trng(gptm: str, inintro: bool, inoutro: bool) -> str:
    match gptm.lower():
        case 'verse': return '<|ss_verse|>'
        case 'chorus': return '<|ss_chorus|>'
        case 'pre-chorus': return '<|ss_prechorus|>'
        case 'bridge': return '<|ss_bridge|>'
        case 'others':
            if inintro: return '<|ss_intro|>'
            elif inoutro: return '<|ss_outro|>'
            else: return '<|ss_interlude|>'
        case _: return ''
timem_rg = r'^\[(\d+).(\d+)(.\d+)?\]'
def _find_tm_idx(rlines: list[str], sfidx: int, sub: str):
    timemark = ''
    idxfind = -1
    if not rlines or sfidx < 0 or not sub:
        return (timemark, idxfind)
    for i in range(sfidx, len(rlines)):
        if sub in rlines[i]:
            idxfind = i
            if m := re.search(timem_rg, rlines[i]):
                timemark = f'[{m.group(1)}:{m.group(2)}'
                ng3 = m.group(3)
                if ng3 and len(ng3) > 1: timemark += f'.{ng3[1:]}'
                timemark += ']'
            break
    return (timemark, idxfind)
def _cnv_lrc4trng(raw_lrc: str, gam_lrc: str) -> tuple[str, str]:
    glines = gam_lrc.strip().splitlines()
    intros = set(); outros = set()
    if grouped_oi := _group_oindices(glines):
        lastli = len(glines) - 1
        goilen = len(grouped_oi)
        if 0 in grouped_oi[0]:
            if goilen == 1 and lastli in grouped_oi[0]:
                intros.add(0)
                outros.add(lastli)
            else:
                intros = grouped_oi[0]
                intros.discard(lastli)
        if goilen > 1 and lastli in grouped_oi[-1]:
            outros = grouped_oi[-1]
            outros.discard(0)
    rlines = raw_lrc.strip().splitlines()
    nextsi = 0
    c4t = ''; d4d = {}
    for i, gl in enumerate(glines):
        parts = re.split(r'\s*\|\s*', gl.strip(), 1)
        if len(parts) == 2 and parts[1]:
            inintro = i in intros
            inoutro = i in outros
            if newsm := _tsf_mss4trng(parts[0], inintro, inoutro):
                newl = newsm + re.sub(r'\d+', '', parts[1])
                c4t += (newl + '\n')
                tm, nextsi = _find_tm_idx(rlines, nextsi, parts[1])
                if tm and -1 < nextsi < len(rlines):
                    d4d[tm] = newl
    c4t = c4t.rstrip()
    c4d = json.dumps(d4d, ensure_ascii=False, separators=(',',':'))
    return (c4t, c4d)
def _core_job(sdict, stst: list[int], lock):
    sid = str(sdict["id"])
    fpct = "{:.3%}".format((stst[0]+stst[1]) / stst[2])
    logger.info(f'cur-progress of sp{stst[3]}: {fpct} (E-{stst[0]}, '
                 f'S-{stst[1]}, T-{stst[2]}), starting job of {sid}')
    rawlrc = sdict['rl']; gamlrc = sdict['gl']
    try:
        c4t, c4d = _cnv_lrc4trng(rawlrc, gamlrc)
        c4tf = f'{dodir}/ali_{sid}_lrc4brotao.txt'
        with open(c4tf, 'w', encoding='utf-8') as f1:
            f1.write(c4t)
        c4df = f'{dodir}/ali_{sid}_lrc4carl.json'
        with open(c4df, 'w', encoding='utf-8') as f2:
            f2.write(c4d)
        if cnx := _try_get_cnx(sid, 64):
            cts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor = cnx.cursor()
            cursor.execute(_st_iou, (cts, c4t, c4d, 'NULL', c4t, c4d, sid))
            cnx.commit()
            cursor.close()
            cnx.close()
        else:
            raise Exception('Can not insert or update to DB')
        logger.info(f'done job for {sid}')
        with lock: stst[1] += 1
    except:
        logger.exception(f'SID-{sid}, EXCEPTION')
        with lock: stst[0] += 1
def _mthp_start(pi: int, songs: list[dict[str, str]]):
    logger.info(f'subprocess{pi} with: {songs[0]["id"]}..{songs[-1]["id"]}')
    ststic = [0, 0, len(songs), pi]
    sclock = threading.Lock()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for song in songs:
            pool.submit(_core_job, song, ststic, sclock)
    logger.info(f'subprocess{pi} finished')
def _mpmt_start(cpu_c):
    logger.info(f'start multiprocesses at pid: {os.getpid()}')
    pp = Pool(cpu_c)
    songs = _load_songs()
    epc = math.floor(len(songs)/cpu_c)
    for i in range(1, cpu_c+1):
        ss = (i-1) * epc
        cpsongs = songs[ss:] if i == cpu_c else songs[ss:i*epc]
        logger.info(f'allocate {len(cpsongs)} songs to subprocess{i}, '
                     f'which first id is {cpsongs[0]["id"]} '
                     f'and last id is {cpsongs[-1]["id"]}')
        pp.apply_async(_mthp_start, args=(i, cpsongs,))
    pp.close()
    pp.join()
    logger.info('all multiprocesses completed')
if __name__ == "__main__":
    cc = cpu_count()
    ucc = 1 if cc < 1 else cc
    logger.info(f'try to use {ucc} cpu-cores on machine{msn}.')
    _mpmt_start(ucc)
    print("Nothing bad happen for running ps, check detail log.")
    logger.info(f'machine{msn} has done multiprocess jobs.')