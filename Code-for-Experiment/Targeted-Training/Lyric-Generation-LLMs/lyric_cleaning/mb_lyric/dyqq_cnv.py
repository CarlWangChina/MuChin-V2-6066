import json
import os
import re
from datetime import datetime
from glob import glob
import mysql.connector
from dotenv import find_dotenv, load_dotenv

fs = glob('/export/efs/data/dy+qq1w/**/music_message.json', recursive=True)
raw_data: dict[str, str] = {}
for f in fs:
    with open(f, 'r', encoding='utf-8') as f:
        for d in json.load(f):
            id = d["id"]
            lrc = d["lyric"]
            if id is not None and len(id) > 0 and \
               lrc is not None and len(lrc) > 0:
                raw_data[id] = lrc

with open('/export/efs/projects/mb-lyric/temp/ori_231226112506.json', 'r', encoding='utf-8') as f:
    llm_trn_list: list[dict] = json.load(f)

load_dotenv(find_dotenv(), override=True)
_dbconfig1 = {
    "host": os.getenv('DB_HOST'),
    "port": os.getenv('DB_PORT'),
    "user": os.getenv('DB_USER'),
    "password": os.getenv('DB_PSWD'),
    "database": "music-annotation",
}
_cnxp = mysql.connector.pooling.MySQLConnectionPool(
    pool_name="mypool_e",
    pool_size=32,
    pool_reset_session=False,
    **_dbconfig1
)
cnx = _cnxp.get_connection()
cursor = cnx.cursor()
#cursor.execute(f)  #This line was incomplete and needs context to fix.
idd = {t[0]: t[1] for t in cursor.fetchall()}
cursor.close()
cnx.close()

_others_rg = r'(?i)^other\s*\|\s*'

def _group_oindices(glines: list[str]) -> list[set[int]]:
    result = []
    cgroup = []
    for i, line in enumerate(glines):
        if re.match(_others_rg, line.lstrip()):
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
        case 'intro' | 'prelude':
            return '<|ss_intro|>'
        case 'interlude':
            return '<|ss_interlude|>'
        case 'outro' | 'ending':
            return '<|ss_outro|>'
        case 'verse':
            return '<|ss_verse|>'
        case 'chorus':
            return '<|ss_chorus|>'
        case 'pre-chorus':
            return '<|ss_prechorus|>'
        case 'bridge':
            return '<|ss_bridge|>'
        case 'other':
            if inintro:
                return '<|ss_intro|>'
            elif inoutro:
                return '<|ss_outro|>'
            else:
                return '<|ss_interlude|>'
        case _:
            return ''

_timem_rg = r'^\[(\d+).(\d+)(.\d+)?\]'

def _find_tm_idx(rlines: list[str], sfidx: int, sub: str):
    timemark = ''
    idxfind = -1
    if not rlines or sfidx < 0 or not sub:
        return (timemark, idxfind)
    for i in range(sfidx, len(rlines)):
        if sub in rlines[i]:
            idxfind = i
            if m := re.search(_timem_rg, rlines[i]):
                timemark = f'[{m.group(1)}:{m.group(2)}'
                ng3 = m.group(3)
                if ng3 and len(ng3) > 1:
                    timemark += f'.{ng3[1:]}'
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
                tm, ix = _find_tm_idx(rlines, nextsi, parts[1])
                if tm and -1 < ix < len(rlines):
                    d4d[tm] = newl
                    nextsi = ix + 1
    c4t = c4t.rstrip()
    c4d = json.dumps(d4d, ensure_ascii=False, separators=(',', ':'))
    return (c4t, c4d)

def xxxx(y: str):
    lines = y.splitlines()
    cs = ''
    rt = ''
    for l in lines:
        if l.startswith('('):
            cs = l[1:-2]
            continue
        rt += f"{cs} | {l}\n"
    return rt.rstrip()

_dbconfig2 = {
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
    **_dbconfig2
)

dir4t = '/export/efs/projects/mb-lyric/data/ouput/dqy_c2ppd/lrc4brotao'
dir4d = '/export/efs/projects/mb-lyric/data/ouput/dqy_c2ppd/lrc4carl'

os.makedirs(dir4t, exist_ok=True)
os.makedirs(dir4d, exist_ok=True)

for trn_song in llm_trn_list:
    jbl = xxxx(trn_song['data1'])
    sid = idd.get(trn_song['mid'])
    if not sid:
        continue
    ral = raw_data.get(sid)
    if not ral:
        continue
    c4t, c4d = _cnv_lrc4trng(ral, jbl)
    with open(f'{dir4t}/{sid}.txt', 'w', encoding='utf-8') as f1:
        f1.write(c4t)
    with open(f'{dir4d}/{sid}.json', 'w', encoding='utf-8') as f2:
        f2.write(c4d)
    cts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cnx = _cnxp.get_connection()
    cursor = cnx.cursor()
    #cursor.execute( , (sid, cts, c4t, c4d, 'NULL', c4t, c4d, cts,)) #This line was incomplete and needs context to fix.
    cnx.commit()
    cursor.close()
    cnx.close()