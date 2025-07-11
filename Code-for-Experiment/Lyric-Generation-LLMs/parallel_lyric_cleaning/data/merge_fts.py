import glob
import json
import sys
import time

def merge_fts(fprefix: str):
    path = f'./data/ouput/{fprefix}*_ft.json'
    files = glob.glob(path)
    if len(files) == 0:
        print(f"no file found with prefix: {fprefix}")
        return
    print(f"found {len(files)} files with prefix: {fprefix}")
    print(f"files: {files}")
    results = []
    for f in files:
        with open(f, 'r', encoding="utf-8") as f:
            results.extend(json.load(f))
    with open('./data/ouput/test20by_gpt-4_ftr.json', 'r', encoding="utf-8") as f:
        results.extend(json.load(f))
    timestamp = int(time.time())
    sf = f'./data/ouput/{fprefix}_all_merged_{timestamp}.json'
    with open(sf, 'w', encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"{len(results)} successfully merged into {sf}")

def merge_mfa_unr():
    dirroot = './data/ouput'
    csuffix = '/lrc/washed_4mfa_unreviewed.json'
    files = [f for f in glob.glob(f'{dirroot}{csuffix}') if f.startswith(('dy', 'qq'))]
    results = []
    for fp in files:
        with open(f'{dirroot}/{fp}', 'r', encoding="utf-8") as f:
            results.extend(json.load(f))
    sf = f'{dirroot}/dy+qq1w_w4mfa_all_merged.json'
    with open(sf, 'w', encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"{len(results)} successfully merged into {sf}")

def merge_llm_unr():
    dirroot = './data/ouput'
    csuffix = '/lrc/washed_4llm_unreviewed.json'
    files = [f for f in glob.glob(f'{dirroot}{csuffix}') if f.startswith(('dy', 'qq'))]
    results = []
    for fp in files:
        with open(f'{dirroot}/{fp}', 'r', encoding="utf-8") as f:
            results.extend(json.load(f))
    sf = f'{dirroot}/dy+qq1w_w4llm_all_merged.json'
    with open(sf, 'w', encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"{len(results)} successfully merged into {sf}")

if __name__ == "__main__":
    merge_llm_unr()