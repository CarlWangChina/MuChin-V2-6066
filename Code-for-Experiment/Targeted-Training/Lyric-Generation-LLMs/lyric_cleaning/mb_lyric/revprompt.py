import json
import logging

logging.basicConfig(
    format='%(asctime)s|%(levelname)s|%(process)d-%(thread)d|%(name)s: %(message)s',
    filename='for_tg_l2p_process.log',
    level=logging.DEBUG,
)

import math
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count

from llmrelate import gpt_req, spt_rl2ppt

_logger = logging.getLogger(__name__)

def load_songs() -> list[dict[str, str]]:
    inputfp = './data/ouput/dy+qq1w_w4llm_all_merged.json'
    with open(inputfp, 'r', encoding="utf-8") as f:
        songs = json.load(f)
    return songs

def rev2prompt(sdict, results, rslock, errors, erlock):
    _logger.debug(f'REVERTING SONG {sdict["id"]}, STATE in set: S-{len(results)}, E-{len(errors)}')
    if lur := str(sdict['lur']):
        try:
            rp = gpt_req(spt_rl2ppt, f'[歌词]：{lur}')
            if rp and len(rp) > 64:
                with rslock:
                    results.append({
                        'id': sdict['id'],
                        'rp': rp
                    })
            else:
                with erlock:
                    errors.append({
                        'id': sdict['id'],
                        'er': f'TOO SHORT rp: {rp}'
                    })
        except:
            ft = traceback.format_exc()
            with erlock:
                errors.append({
                    'id': sdict['id'],
                    'er': f'EXCEPTION: {ft}'
                })
    else:
        with erlock:
            errors.append({
                'id': sdict['id'],
                'er': 'EMPTY SOURCE lur'
            })

def mthp_start(pi: int, songs: list[dict[str, str]]):
    _logger.info(f'subprocess{pi} with: {songs[0]["id"]}..{songs[-1]["id"]}')
    rl = threading.Lock()
    el = threading.Lock()
    ra = []
    ea = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for s in songs:
            pool.submit(rev2prompt, s, ra, rl, ea, el)
    sf = f'./data/ouput/for_tg_l2p_mp{pi}.json'
    with open(sf, 'w', encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False)
    _logger.info(f"subprocess{pi} saved {len(ra)} in result file: {sf}")
    if len(ea) > 0:
        ef = f'./data/ouput/for_tg_l2p_mp{pi}_errs.json'
        with open(ef, 'w', encoding="utf-8") as f:
            json.dump(ea, f, ensure_ascii=False)
        _logger.warning(f"subprocess{pi} has {len(ea)} errors, see: {ef}")
    _logger.info(f'subprocess{pi} finished')

def mpmt_start():
    _logger.debug(f'start multiprocessing at pid: {os.getpid()}')
    cc = cpu_count()
    ucc = 1 if cc <= 1 else math.floor(cc/2)
    _logger.info(f'try using {ucc} from total {cc} of device cpu-cores.')
    pp = Pool(ucc)
    songs = load_songs()[3000:]
    epc = math.floor(len(songs)/ucc)
    for i in range(1, ucc+1):
        ss = (i-1) * epc
        cpsongs = songs[ss:] if i == ucc else songs[ss:i*epc]
        _logger.debug(f'allocate {len(cpsongs)} songs to subprocess{i}, which first id is {cpsongs[0]["id"]} and last id is {cpsongs[-1]["id"]}')
        pp.apply_async(mthp_start, args=(i, cpsongs,))
    pp.close()
    pp.join()
    _logger.debug('all multiprocesses completed')

def spst_start():
    songs = load_songs()
    errors = []
    results = []
    for i, s in enumerate(songs):
        if i > 2999:
            print('stop at index 3000')
            break
        if lur := str(s['lur']):
            try:
                rp = gpt_req(spt_rl2ppt, f'[歌词]：{lur}')
                if rp and len(rp) > 64:
                    results.append({
                        'id': s['id'],
                        'rp': rp
                    })
                else:
                    errors.append({
                        'id': s['id'],
                        'er': f'TOO SHORT rp: {rp}'
                    })
            except:
                ft = traceback.format_exc()
                errors.append({
                    'id': s['id'],
                    'er': f'EXCEPTION: {ft}'
                })
        else:
            errors.append({
                'id': s['id'],
                'er': 'EMPTY SOURCE lur'
            })
    timestamp = int(time.time())
    sf = f'./data/ouput/for_tg_{timestamp}_part1.json'
    with open(sf, 'w', encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"{len(results)} successfully saved in {sf}")
    if len(errors) > 0:
        ef = f'./data/ouput/for_tg_{timestamp}_part1_errs.json'
        with open(ef, 'w', encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False)
        print(f"{len(errors)} error logs recorded at {ef}")

if __name__ == "__main__":
    mpmt_start()