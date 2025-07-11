import argparse
import json
import os
import time
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", metavar='C', type=str, help="path to the checkpoint files root.")
parser.add_argument("-b", "--base-model", type=str, default=None, help="path to the base model root.")
parser.add_argument("-r", "--io-root", type=str, required=True, help="path root for infer's input/output files.")
parser.add_argument("-f", "--file-name", type=str, required=True, help="name of the input file need to be inferred.")
parser.add_argument("-x", "--times-each", type=int, default=1, choices=range(1, 21), help="number of times infer for each pair.")
args = parser.parse_args()

ppt_tpl4bm_1shot = ""
ppt_tpl4fm_0shot = ""

if args.base_model is None:
    args.base_model = args.checkpoint
if args.checkpoint == args.base_model:
    from transformers import AutoModelForCausalLM as AutoModel4CLM
    prompt_tpl = ppt_tpl4bm_1shot
else:
    from peft import AutoPeftModelForCausalLM as AutoModel4CLM
    prompt_tpl = ppt_tpl4fm_0shot

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
model = AutoModel4CLM.from_pretrained(args.checkpoint, device_map="auto", trust_remote_code=True).eval()

input_file = f"{args.io_root}/{args.file_name}"
with open(input_file, "r", encoding="utf-8") as fi:
    theme_msstr_pairs = json.load(fi)

dtnow_fmt = datetime.now().strftime('%y%m%d%H%M')
ckpt_name = os.path.basename(args.checkpoint.rstrip('/'))
ouput_dir = f"{args.io_root}/{ckpt_name}/{args.file_name.split('.')[0]}/ntie{args.times_each}_{dtnow_fmt}"
os.makedirs(ouput_dir, exist_ok=True)

for tm_pair in theme_msstr_pairs:
    opp_common = f"pid_{tm_pair['id']:03d}"
    tf_dir = f"{ouput_dir}/{opp_common}"
    os.makedirs(tf_dir, exist_ok=True)
    with open(f"{tf_dir}/p_theme.txt", "w", encoding="utf-8") as fo:
        fo.write(tm_pair['theme'])
    usr_input = prompt_tpl % (tm_pair['theme'], tm_pair['msstr'])
    results = []
    for i in range(args.times_each):
        infer_stime = time.time()
        lrc, _ = model.chat(tokenizer, usr_input, history=None)
        infer_etime = time.time()
        results.append({
            "gen_lrc": lrc,
            "seccost": infer_etime - infer_stime,
            "obj_evl": None,
            "sbj_evl": None,
        })
        txt_file = f"{tf_dir}/lrc_{i+1:02d}.txt"
        with open(txt_file, "w", encoding="utf-8") as fo:
            fo.write(lrc)
    op_dict = {
        "done_ts": int(time.time()),
        "o_msstr": tm_pair['msstr'],
        "o_theme": tm_pair['theme'],
        "results": results
    }
    op_json = f"{ouput_dir}/{opp_common}_all.json"
    with open(op_json, "w", encoding="utf-8") as fo:
        json.dump(op_dict, fo, indent=2, ensure_ascii=False)