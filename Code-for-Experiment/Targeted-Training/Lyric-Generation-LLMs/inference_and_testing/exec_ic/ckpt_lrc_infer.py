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

ppt_tpl4bm_1shot = """你将扮演一位流行音乐领域的专业作词人，根据[歌词主旨]和[模板结构]对应生成一篇严格符合要求的歌词，比如：  [模板结构]假设是    "cccccccccccc\nccccccccR\ncccccccccccR\ncccccccccR\ncccccccccccR\nccccccccc\ncccccccccccR\nccccccccc\ncccccccccccc\nccccccccR\nccccccccccR\nccccccccR\ncccccccccccR\nccccccccR\ncccccccccccR\nccccccccR"  [歌词主旨]假设是    "我想表达回忆、遗憾和时间流逝的深刻感慨。试图忘记过去的情感纠葛，但又无法真正释怀。对往昔美好时光的记忆也暗示着这些记忆带来的痛苦，希望接受命运并随着时间慢慢淡忘过去。"  则最终输出的歌词应该类似于    "就这样忘记吧怎么能忘记呢\n墨绿色的纠缠你的他\n窗前流淌的歌枕上开过的花\n岁月的风带它去了哪啊\n就这样忘记吧怎么能忘记呢\n昏黄色的深情你的他\n指尖燃起的火喉头咽下的涩\n瞳孔里的星辰在坠落\n总有些遗憾吗总有些遗憾吧\n光阴它让纯粹蒙了灰\n如此蒂固根深又摇摇欲坠\n倒影中的轮廓他是谁\n你也是这样吗你也是这样吧\n目送了太久忘了出发\n说不出是亏欠等不到是回答\n就这样老去吧老去吧"即在符合[歌词主旨]要求的前提下，严格按照[模板结构]中给出的格式生成歌词，其中表示换行的'\n'保持不变，每一个'c'或'R'都代表着需要替换生成的一个中文字，但注意'R'不同与'c'的是，所有'R'的替换生成字都必须属于同一种押韵！请**严格**按照以上要求只给出最终结果，不能输出除结果以外的任何其他信息。歌词主旨：%s模板结构：%s"""
ppt_tpl4fm_0shot = """你将扮演一位流行音乐领域的专业作词人，根据[歌词主旨]和[模板结构]对应生成一篇歌词，[模板结构]里的'\n'表示换行，空格' '表示停顿，这些信息须在输出时保持不变，其余的每一个'c'或'R'都代表着需要替换生成的一个中文字，但注意'R'不同于'c'的是，所有'R'的替换生成字都必须属于同一种押韵！请**严格**按照以上要求只给出最终结果，不能输出除结果以外的任何其他信息。歌词主旨：%s模板结构：%s"""

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