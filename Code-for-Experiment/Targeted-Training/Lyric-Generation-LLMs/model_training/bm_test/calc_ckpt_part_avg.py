import re
import sys
from glob import glob
from pprint import pprint
import numpy as np

root_dir = '/home/carl/mproj/qwen-trn/data/ciio'
tdst_name = sys.argv[1]
ckpt_name = sys.argv[2]
logs = glob(f'{root_dir}/{ckpt_name}/obj_evl_logs/{tdst_name}*/**/pid_*.sr.log', recursive=True)
d1_re_p = r'^phase1 +score: *(\d*\.*\d+),'
d2_1_re_p = r'^phase2.1 +score: *(\d*\.*\d+),'
d2_2_re_p = r'^phase2.2 +score: *(\d*\.*\d+),'
d3_re_p = r'^phase3 +score: *(\d*\.*\d+),'
d4_re_p = r'^phase4 +score: *(\d*\.*\d+),'
de_re_p = r'^extra +score: *(\d*\.*\d+),'
part_scores = {}
for log in logs:
    with open(log, 'r', encoding='utf-8') as f:
        for line in f.readlines()[-7:]:
            if d1m := re.match(d1_re_p, line):
                part_scores.setdefault('d1', [])
                part_scores['d1'].append(float(d1m.group(1)))
            elif d2_1m := re.match(d2_1_re_p, line):
                part_scores.setdefault('d2_1', [])
                part_scores['d2_1'].append(float(d2_1m.group(1)))
            elif d2_2m := re.match(d2_2_re_p, line):
                part_scores.setdefault('d2_2', [])
                part_scores['d2_2'].append(float(d2_2m.group(1)))
            elif d3m := re.match(d3_re_p, line):
                part_scores.setdefault('d3', [])
                part_scores['d3'].append(float(d3m.group(1)))
            elif d4m := re.match(d4_re_p, line):
                part_scores.setdefault('d4', [])
                part_scores['d4'].append(float(d4m.group(1)))
            elif dem := re.match(de_re_p, line):
                part_scores.setdefault('de', [])
                part_scores['de'].append(float(dem.group(1)))
p_stis = {}
for k, v in part_scores.items():
    p_stis[f'{k}.avg'] = np.mean(v)
    p_stis[f'{k}.var'] = np.var(v)
    p_stis[f'{k}.std'] = np.std(v)
pprint(p_stis)