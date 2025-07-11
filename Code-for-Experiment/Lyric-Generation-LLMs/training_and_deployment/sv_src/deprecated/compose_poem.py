import logging
logging.basicConfig(
    format='%(asctime)s|%(levelname)s|%(process)d-%(thread)d|%(name)s: %(message)s',
    level=logging.INFO)
_logger = logging.getLogger(__name__)
import re
import torch
from LAC import LAC
from peft import AutoPeftModelForCausalLM as AutoModelForCausalLM
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers.generation import GenerationConfig
import Code_for_Experiment.Targeted_Training.data_synthesis_tool.scripts.convert_dyqy

def init_all_models():
    global tknzer_libai, model_libai, lac, tknzer_bert, model_bert
    tknzer_libai = AutoTokenizer.from_pretrained(
        '/data/shared/Qwen/base-model',
        trust_remote_code=True,
        resume_download=True)
    model_libai = AutoModelForCausalLM.from_pretrained(
        '/data/shared/Qwen/final-ckpt/',
        device_map='auto',
        trust_remote_code=True,
        resume_download=True).eval()
    model_libai.generation_config = GenerationConfig.from_pretrained(
        '/data/shared/Qwen/final-ckpt/',
        trust_remote_code=True,
        resume_download=True)
    lac = LAC(mode='rank')
    tknzer_bert = AutoTokenizer.from_pretrained(
        '/data/shared/Qwen/bert-bc')
    model_bert = AutoModelForMaskedLM.from_pretrained(
        '/data/shared/Qwen/bert-bc')

def predict_masked(masked_lrc: str):
    _logger.info(f"----BERT Predicting---- input: {masked_lrc}")
    global tknzer_bert, model_bert
    inputs = tknzer_bert(masked_lrc, return_tensors="pt")
    with torch.no_grad():
        logits = model_bert(**inputs).logits
        mask_tk_idx = (
            inputs.input_ids == tknzer_bert.mask_token_id
        )[0].nonzero(as_tuple=True)[0]
        mask_logits = logits[0, mask_tk_idx]
        filtered_logits = torch.full_like(mask_logits, float('-inf'))
        filtered_logits[:, 672:7993] = mask_logits[:, 672:7993]
        predicted_token_id = filtered_logits.argmax(axis=-1)
    result = tknzer_bert.decode(predicted_token_id)
    _logger.info(f"----BERT Predicting---- ouput: {result}")
    return re.sub(r'[\W_]+', '', zhcvt(result, 'zh-cn'))

def cutdown_word(gl_line: str, dec: int):
    _logger.info(f"----LAC  Downsizing---- input: {gl_line}")
    assert 0 < dec < len(gl_line), "Invalid number."
    global lac
    lac_r = lac.run(gl_line)
    words = lac_r[0]
    r_zip = zip(
        words,
        lac_r[1],
        lac_r[2],
        range(len(words)),
    )
    rank_sorted = sorted(r_zip, key=lambda t: t[2])
    remain = dec
    idx2del = []
    for w, _, r, i in rank_sorted:
        if i == len(words) - 1: continue
        temp = remain - len(w)
        if r > 1 or remain <= 0 or temp < 0:
            break
        else:
            idx2del.append(i)
            remain = temp
    result = str(''.join([
        w for i, w in enumerate(words) if i not in idx2del]))
    if remain: result = result[remain:]
    _logger.info(f"----LAC  Downsizing---- ouput: {result}")
    return result

def ajust_eline_words(
    gl_line: str,
    cr_line: str,
    a_lines: "list[str]",
    g_lines: "list[str]",
    cur_idx: int,
):
    gl_len = len(gl_line)
    cl_len = len(cr_line)
    if gl_len == cl_len: return gl_line
    elif gl_len > cl_len:
        diff = gl_len - cl_len
        if (diff / cl_len > 0.5 or diff > 3) and g_lines:
            candidate_ids = [i for i, s in enumerate(g_lines)
                if len(s) >= cl_len]
            min_delta_idx = min(candidate_ids,
                key=lambda j: len(g_lines[j]) - cl_len)
            pms_l = g_lines[min_delta_idx]
            n_extra = len(pms_l) - cl_len
            if n_extra == 0:
                return pms_l
            elif n_extra < diff:
                new_l = cutdown_word(pms_l, n_extra)
            else:
                new_l = cutdown_word(gl_line, diff)
        else:
            new_l = cutdown_word(gl_line, diff)
    else:
        diff = cl_len - gl_len
        mask_l = '[MASK]'*diff + gl_line
        wc_sum = diff + gl_len
        pl_idx = cur_idx - 1
        nl_idx = cur_idx + 1
        arange = range(len(a_lines))
        grange = range(len(g_lines))
        while diff / wc_sum > 0.0625:
            if pl_idx in arange:
                pl = a_lines[pl_idx]
                mask_l = pl + '，' + mask_l
                wc_sum += len(pl)
                pl_idx -= 1
            elif nl_idx in grange:
                nl = g_lines[nl_idx]
                mask_l = mask_l + '，' + nl
                wc_sum += len(nl)
                nl_idx += 1
            else:
                break
        new_l = predict_masked(mask_l) + gl_line
    return new_l

def force_align(g_lrc: str, s_ccr: str) -> str:
    glines = [l for l in g_lrc.splitlines() if l.strip()]
    slines = [l for l in s_ccr.splitlines() if l.strip()]
    glen = len(glines)
    slen = len(slines)
    assert glen * slen > 0, "No valid lines found in input."
    if glen > slen:
        glines = glines[:slen]
    elif glen < slen:
        diff = slen - glen
        glines += glines[-diff:]
    aligned = []
    for i, (gl, sl) in enumerate(zip(glines, slines)):
        aligned.append(
            ajust_eline_words(gl, sl, aligned, glines, i))
    return '\n'.join(aligned)

ppt_tpl_0shot = """
## 创作歌词
### 主题：%s
### 模板：%s
"""

def compose_poem(mid: str, usr_ppt: str, ccr_tpl: str):
    ccr_ccd = ccr_tpl.strip().replace('C', 'c').replace('r', 'R')
    input = ppt_tpl_0shot % (usr_ppt.strip(), ccr_ccd)
    global tknzer_libai, model_libai
    _logger.info(f"mid-{mid} | Formatted usr input:\n{input}")
    lyric, _ = model_libai.chat(tknzer_libai, input, history=None)
    _logger.info(f"mid-{mid} | Generated raw lyric:\n{lyric}")
    lyric = re.sub(r'[^\u4e00-\u9fff\s]', '', lyric.strip())
    lyric = re.sub(r'[ \t\r\f\v]+', ' ', lyric)
    lyric = re.sub(r'\n{2,}', '\n', lyric)
    lyric = str(zhcvt(lyric, 'zh-cn'))
    try:
        return force_align(lyric, ccr_ccd)
    except:
        _logger.exception(f"mid-{mid} !! Lyric force-alignment failed.")
        _logger.warning(f"mid-{mid} ! Returning raw generated lyric.")
        return lyric

if __name__ == '__main__':
    init_all_models()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--purport", type=str, required=True,
        help="the purport or theme to generate the lyric for.")
    parser.add_argument("-s", "--structure", type=str, required=True,
        help="structure template for lyric, in 'cccR' format.")
    parser.add_argument("-i", "--identity", type=str, default="test",
        help="the identity of music or task, for logging purpose.")
    args = parser.parse_args()
    print(f'0-s = {args.structure}')
    structure = args.structure.replace('\\n', '\n')
    print(f'1-s = {structure}')
    lyric_poem = compose_poem(args.identity, args.purport, structure)
    print(lyric_poem)