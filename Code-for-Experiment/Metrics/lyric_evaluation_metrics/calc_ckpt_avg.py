import json
import os
from glob import glob
import numpy as np
from Code_for_Experiment.Targeted_Training.Lyric_Generation_LLMs.model_training.bm_test.glrc_obj_eval import scoring

root_dir = '/home/carl/mproj/qwen-trn/data/ciio/'
jf_paths = glob(f'{root_dir}*/**/pid_*_all.json', recursive=True)

def save_mid_records(log_str, df_html, path_parts: "list[str]"):
    log_root_dir = f'{root_dir}/{path_parts[0]}/obj_evl_logs'
    log_dir = f'{log_root_dir}/{"@".join(path_parts[1:-1])}'
    os.makedirs(log_dir, exist_ok=True)
    for i in range(1):
        fn_cp = f'{path_parts[-1]}${(i+1):02d}'
        dl_fn = f'{fn_cp}.sr.log'
        dh_fn = f'{fn_cp}.df.html'
        with open(f'{log_dir}/{dl_fn}', 'w') as f:
            f.write(log_str)
        with open(f'{log_dir}/{dh_fn}', 'w') as f:
            f.write(df_html)

scores_dict: "dict[str, float]" = {}
for json_file in jf_paths:
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data: dict = json.load(f)
    sub_path_parts = json_file[len(root_dir):].split('/')
    sub_path_parts[-1] = sub_path_parts[-1].rstrip('.json')
    o_msstr = json_data.get('o_msstr', '')
    results: "list[dict]" = json_data.get('results', [])
    changed = False
    for i, result in enumerate(results):
        ori_oe_score = result.get('obj_evl', 0.0)
        if not isinstance(ori_oe_score, float):
            ori_oe_score = 0.0
        ori_se_score = result.get('sbj_evl', 0.0)
        if not isinstance(ori_se_score, float):
            ori_se_score = 0.0
        gen_lrc = result.get('gen_lrc', '')
        oe_score, oe_log, oe_html = scoring(gen_lrc, o_msstr)
        if ori_oe_score != oe_score:
            result['obj_evl'] = oe_score
            changed = True
            ori_oe_score = oe_score
        save_mid_records(oe_log, oe_html, sub_path_parts)
        sd_key = "@".join(sub_path_parts) + f'${(i+1):02d}'
        scores_dict[sd_key] = ori_oe_score + ori_se_score
    if changed:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

with open(f'{root_dir}all_scores.json', 'w', encoding='utf-8') as f:
    json.dump(scores_dict, f, indent=2, ensure_ascii=False)

inflated_sld: "dict[str, list[float]]" = {}

def rec_calc_avg(ul_obj: dict, score: float, key_n: str, key_p: str = ''):
    "Recursively calculate the average score for each dimension."
    if '@' not in key_n:
        c_key = key_n.split('$', 1)[0]
        obj: dict = ul_obj.setdefault(c_key, {})
        scores: list = obj.setdefault('scores', [])
        scores.append(score)
        obj["__AVG__"] = np.mean(scores)
        obj["__VAR__"] = np.var(scores)
        obj["__STD__"] = np.std(scores)
    else:
        parts = key_n.split('@', 1)
        c_key = parts[0]
        isd_key = key_p + c_key
        dsl = inflated_sld.setdefault(isd_key, [])
        dsl.append(score)
        obj: dict = ul_obj.setdefault(c_key, {})
        obj["__AVG__"] = np.mean(dsl)
        obj["__VAR__"] = np.var(dsl)
        obj["__STD__"] = np.std(dsl)
        rec_calc_avg(obj, score, parts[1], isd_key)

multi_d_avgs = {}
for key, score in scores_dict.items():
    rec_calc_avg(multi_d_avgs, score, key)

with open(f'{root_dir}all_d_avgs.json', 'w', encoding='utf-8') as f:
    json.dump(multi_d_avgs, f, indent=2, ensure_ascii=False)