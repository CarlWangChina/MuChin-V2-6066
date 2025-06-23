import asyncio
import json
import logging
import os
import sys
import threading
import time
import traceback
import openai
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")
openaimodel = os.getenv("OPENAI_MODEL_N")
from mb_lyric.llmrelate import sysprompt as syscontent

results = []
errors = []
pcount = 0
lock = threading.Lock()

def cur_fmt_time():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

def loop_check_prog(end: int, interval: int = 20):
    while True:
        with lock:
            c = pcount
        if c >= end:
            break
        print(f"{cur_fmt_time()} progress: {c}/{end}")
        time.sleep(interval)

def verify_result(dict_):
    "raise exception if `dict` is not valid"
    if 'prompt' not in dict_:
        raise Exception("The 'prompt' part is missing!")
    if len(dict_['completion']) < 30:
        raise Exception("The 'completion' is too short (<30).")
    if dict_['completion'].find('[') > -1:
        raise Exception("The 'completion' still contains LRC tags.")

async def analyse2json(lrc: str):
    "start analyse raw `lrc` to json object by openai request"
    respctt = ''
    try:
        response = openai.ChatCompletion.create(
            model=openaimodel,
            messages=[
                {"role": "system", "content": syscontent},
                {"role": "user", "content": f"[歌词]:{lrc}"}
            ],
        )
        respctt = response['choices'][0]['message']['content']
        result = respctt[respctt.find('{'):respctt.rfind('}')+1]
        dict_ = json.loads(result, strict=False)
        verify_result(dict_)
        results.append(dict_)
    except Exception as e:
        ft = traceback.format_exc()
        es = f"Error: {ft}\nSource LRC:\n{lrc}\nResp Content:\n{respctt}"
        errors.append(es)
    finally:
        with lock:
            global pcount
            pcount += 1

async def app_main_job(lyrics: list, fprefix: str):
    "create and parallelly run tasks of `analyse2json`, save results to file"
    tasks = [asyncio.create_task(analyse2json(lrc)) for lrc in lyrics]
    print(f"tasks count: {len(tasks)}")
    print(f"tasks start at: {cur_fmt_time()}")
    await asyncio.gather(*tasks)
    timestamp = int(time.time())
    sf = f'./data/ouput/{fprefix}_{openaimodel}_{timestamp}_ft.json'
    with open(sf, 'w', encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"{len(results)} successfully saved in {sf}")
    if len(errors) > 0:
        ef = f'./data/ouput/log/{fprefix}_{openaimodel}_{timestamp}'
        with open(ef, 'w', encoding="utf-8") as f:
            f.write(f'\n{"-" * 80}\n'.join(errors))
        print(f"{len(errors)} error logs recorded at {ef}")
    print(f"tasks finished: {cur_fmt_time()}")

def main_async_wrap(lrcs, fprefix):
    asyncio.run(app_main_job(lrcs, fprefix))

def load_lyrics(fname: str):
    "load lyric list from json file by the `fname`"
    path = fname
    if not os.path.exists(path):
        path = f'./data/input/{fname}'
        if not os.path.isfile(path):
            raise Exception(f"file not found: neither in {fname} nor in {path}")
    with open(path, 'r', encoding="utf-8") as f:
        musicinfos = json.load(f)
    return [str(mi['lyric']) for mi in musicinfos]

if __name__ == "__main__":
    inputfp = f'./data/input/{sys.argv[1]}'
    fnwithe = inputfp.rsplit('/', 1)[1]
    fnwoute = fnwithe.rsplit('.', 1)[0]
    with open(inputfp, 'r', encoding="utf-8") as f:
        songinfos = json.load(f)
    from Code_for_Experiment.Targeted_Training.Lyric_Generation_LLMs.lyric_cleaning.async_fwr import FwThread
    from Code_for_Experiment.Targeted_Training.Lyric_Generation_LLMs.lyric_cleaning.mb_lyric.lrcleaner import MsMuscle
    csvsdir = f'./data/ouput/{fnwoute}/rec/cln'
    mfaopfn = f'./data/ouput/{fnwoute}/lrc/'
    os.makedirs(mfaopfn, exist_ok=True)
    os.makedirs(csvsdir, exist_ok=True)
    arr = []
    mm = MsMuscle()
    for si in songinfos:
        slyric = str(si['lyric']).strip()
        if not slyric: continue
        songid = str(si['id']).strip()
        if not songid:
            fid = str(round(time.time()*1000))[:9]
            songid = f'FID_{fid}_{si["name"]}'
        cleanresult = mm.wash_lrc_4llm(slyric)
        cl = cleanresult[0]
        if cl: arr.append({'id': songid, 'lur': cl})
        csvfn = f'{csvsdir}/{songid}-llm.csv'
        FwThread(csvfn, cleanresult[1]).start()
    mfaopfn += 'washed_4llm_unreviewed.json'
    with open(mfaopfn, 'w', encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False)
    print('Done!')