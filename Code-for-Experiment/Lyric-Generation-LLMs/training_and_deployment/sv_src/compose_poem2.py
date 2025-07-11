import logging
logging.basicConfig(
    format='%(asctime)s|%(levelname)s|%(process)d-%(thread)d|%(name)s: %(message)s',
    level=logging.INFO)
_logger = logging.getLogger(__name__)
import math
import random
import re
import time
from dataclasses import dataclass
import jieba
import torch
import torch.nn.functional as tnf
from peft import AutoPeftModelForCausalLM as AutoModelForCausalLM
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
from transformers.generation import GenerationConfig
import Code_for_Experiment.Targeted_Training.data_synthesis_tool.scripts.convert_dyqy

def _init_bert_only(p: str='/data/shared/Qwen/bert-bc'):
    global tknzer_bert, model_bert, model_bert4m, device_b
    tknzer_bert = AutoTokenizer.from_pretrained(p)
    device_b = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_bert = AutoModel.from_pretrained(p).to(device_b)
    model_bert4m = AutoModelForMaskedLM.from_pretrained(p).to(device_b)

def init_all_models():
    global tknzer_libai, model_libai
    tknzer_libai = AutoTokenizer.from_pretrained(
        '/data/shared/Qwen/base-model',
        trust_remote_code=True,
        resume_download=True,
    )
    model_libai = AutoModelForCausalLM.from_pretrained(
        '/data/shared/Qwen/final-ckpt',
        device_map='auto',
        trust_remote_code=True,
        resume_download=True,
    ).eval()
    model_libai.generation_config = GenerationConfig.from_pretrained(
        '/data/shared/Qwen/final-ckpt',
        trust_remote_code=True,
        resume_download=True,
    )
    _init_bert_only()

@dataclass
class Sentence:
    ltext: str
    "Line text of this sentence"
    index: int
    "Index in the generated lines"
    ut_cr: int = 0
    "Used times for copy replace"
    ut_ct: int = 0
    "Used times by copy and trim"
    ut_cf: int = 0
    "Used times by copy and fill"
    def __eq__(self, other):
        if not isinstance(other, Sentence): return False
        return self.ltext == other.ltext \
            and self.index == other.index \
            and self.ut_cr == other.ut_cr \
            and self.ut_ct == other.ut_ct \
            and self.ut_cf == other.ut_cf
    def __hash__(self):
        return hash((self.ltext, self.index, self.ut_cr,
            self.ut_ct, self.ut_cf))

@dataclass
class SameLenCs:
    stc_l: "list[Sentence]"
    "Sentence list."
    c_len: int
    "Common length."
    ltu_i: int = -1
    "Last time used index."

@dataclass
class ComposPoem:
    l_glt: "list[str]"
    "Listed of generated line text."
    m_l2s: "dict[int, SameLenCs]"
    "Mapper, length -> SameLenCs."
    t_len: int
    "Total listed (l_glt) length."

def sample_reduce(ss_list: "list[str]", each_ls=0, threshold=512):
    if not ss_list: return ss_list
    if each_ls < 1 and not (each_ls := len(ss_list[0])):
        return ss_list
    max_len = threshold // each_ls
    if max_len >= len(ss_list): return ss_list
    return random.sample(ss_list, max_len)

def all_possi_prune(line: str, n_d: int, keep_lc: bool):
    if len(line) <= n_d or n_d <= 0:
        return [line]
    last_c = ''
    if keep_lc:
        last_c = line[-1]
        line = line[:-1]
    if len(line) > 16:
        _logger.warning('Too long line, remain last 16 chars.')
        line = line[-16:]
    result = []
    for i in range(1 << len(line)):
        if bin(i).count('1') == len(line) - n_d:
            subset = ''.join([line[j] for j in range(len(line))
                              if i & (1 << j)])
            result.append(subset + last_c)
        if len(result) > 255:
            _logger.warning('Possibilities OF, use first 256.')
            break
    return result

def trim_cutdown(line: str, n_d: int, kp_lc: bool=True):
    _ST = time.perf_counter()
    global tknzer_bert, model_bert, device_b
    bias_ipt = tknzer_bert(line, return_tensors="pt").to(device_b)
    prune_ls = all_possi_prune(line, n_d, kp_lc)
    prune_ls = sample_reduce(prune_ls)
    encodi_d = {k: v.repeat(len(prune_ls), 1) for k, v in bias_ipt.items()}
    comp_ipt = tknzer_bert(prune_ls, return_tensors="pt",
        padding=True, truncation=True).to(device_b)
    for key in comp_ipt.keys():
        encodi_d[key] = torch.cat(
            (encodi_d[key], comp_ipt[key]), dim=1).to(device_b)
    with torch.no_grad():
        outputs = model_bert(**encodi_d)
    embeddings = outputs.last_hidden_state
    bias_embdi = embeddings[:, 0, :]
    comp_embdi = embeddings[:, 1, :]
    cos_scores = tnf.cosine_similarity(bias_embdi, comp_embdi, dim=1)
    smax_index = cos_scores.argmax().item()
    _ET = time.perf_counter()
    _logger.info(
        f"TSC: {_ET-_ST:.2f}s, trim_cutdown({line}, {n_d}, {kp_lc})")
    return prune_ls[int(smax_index)]

def predict_for_best(masked_wctx_ll: "list[str]", mn_e: int):
    global tknzer_bert, model_bert4m, device_b
    inputs = tknzer_bert(masked_wctx_ll, return_tensors="pt",
        padding=True, truncation=True).to(device_b)
    with torch.no_grad():
        logits = model_bert4m(**inputs).logits
    mask_indices = inputs.input_ids == tknzer_bert.mask_token_id
    mask_logits = logits[mask_indices]
    filtered_logits = torch.full_like(mask_logits, float('-inf'))
    filtered_logits[:, 672:7993] = mask_logits[:, 672:7993]
    predicted_tkids = filtered_logits.argmax(dim=-1)
    pred_words = tknzer_bert.batch_decode(
        predicted_tkids, skip_special_tokens=True)
    fl_softmax = tnf.softmax(filtered_logits, dim=-1)
    cfd_scores = fl_softmax.max(dim=-1).values.tolist()
    if mn_e == 1:
        prediction = pred_words
        cfd_sc_sum = cfd_scores
    else:
        prediction = [''.join(pred_words[i:i+mn_e])
                     for i in range(0, len(pred_words), mn_e)]
        cfd_sc_sum = [float(sum(cfd_scores[i:i+mn_e]))
                     for i in range(0, len(cfd_scores), mn_e)]
    sc_max_idx = max(enumerate(cfd_sc_sum), key=lambda t: t[1])[0]
    return prediction[sc_max_idx], sc_max_idx

def assemble_ctx_4mask(cur_i: int, glines: "list[str]"):
    ctx_ps, ctx_ns, join_pm = '', '', '\n'
    wc_sum = 0
    grange = range(len(glines))
    pl_idx = cur_i - 1
    while wc_sum < 32:
        if pl_idx not in grange:
            break
        else:
            pl = glines[pl_idx]
            ctx_ps += pl + join_pm
            wc_sum += len(pl)
            pl_idx -= 1
    nl_idx = cur_i + 1
    while wc_sum < 56:
        if nl_idx in grange:
            nl = glines[nl_idx]
            ctx_ns += join_pm + nl
            wc_sum += len(nl)
            nl_idx += 1
        else:
            break
    return ctx_ps, ctx_ns

def recrs_setup_mkls(phrases: "list[str]", n: int, start=0):
    if n == 0: return [''.join(phrases)]
    results: list[str] = []
    for i in range(start, len(phrases)):
        new_l = phrases[:]
        new_l.insert(i, '[MASK]')
        results.extend(recrs_setup_mkls(new_l, n-1, i+1))
    return results

def fill_inflate(cur_i: int, n: int, lines: "list[str]"):
    _ST = time.perf_counter()
    line = lines[cur_i]
    ctx_ps, ctx_ns = assemble_ctx_4mask(cur_i, lines)
    es_mlwc = len(ctx_ps) + len(ctx_ns)
    phrases = list(jieba.cut(line))
    maskedl = recrs_setup_mkls(phrases, n)
    es_mlwc += len(maskedl[0])
    maskedl = sample_reduce(maskedl, each_ls=es_mlwc)
    mls_wctx = [ctx_ps+ml+ctx_ns for ml in maskedl]
    best_p, best_i = predict_for_best(mls_wctx, n)
    best_p = re.sub(r'[\W_]+', '', zhcvt(best_p, 'zh-cn'))
    best_t = maskedl[best_i]
    if (mn := best_t.count('[MASK]') - len(best_p)) > 0:
        best_p = best_p + ''.join(random.choices('啊呐啦噢', k=mn))
    _ET = time.perf_counter()
    _logger.info(
        f"TSC: {_ET-_ST:.2f}s, fill_inflate({line}, {n}, _)")
    return best_t.replace('[MASK]', '{}').format(*best_p)

def create_newl(at_i: int, n: int, lines: "list[str]"):
    _ST = time.perf_counter()
    ctx_ps, ctx_ns = assemble_ctx_4mask(at_i, lines)
    mls_wctx = [ctx_ps + '[MASK]'*n + ctx_ns]
    best_p, _ = predict_for_best(mls_wctx, n)
    best_p = re.sub(r'[\W_]+', '', zhcvt(best_p, 'zh-cn'))
    if (mn := n - len(best_p)) > 0:
        best_p = best_p + ''.join(random.choices('啊呐啦噢', k=mn))
    _ET = time.perf_counter()
    _logger.info(
        f"TSC: {_ET-_ST:.2f}s, create_newl({at_i}, {n}, _)")
    return best_p

def exp_score(weight: float, sigma: float, n: int):
    return weight * math.exp(-(n ** 2) / (2 * sigma ** 2))

def log_score(weight: float, alpha: float, n: int):
    return -alpha * math.log1p(abs(n)) + weight

def lin_score(weight: float, dnntr: float, n: int):
    return weight * (1 - (n / dnntr))

def distance_of(anchor_i: int, index: int, l_len: int):
    if l_len < 1: return 0
    delta = index - anchor_i
    return delta % l_len

WEIGHTS = (50.0, 20.0, 15.0, 15.0)

def multi_dimen_score(
    st: Sentence,
    dv_m2sl: int,
    gll_len: int,
    cur_idx: int,
    ltu_idx: int,
):
    score = 0.0
    score += exp_score(WEIGHTS[0], 2.0, dv_m2sl)
    if dv_m2sl > 0: score *= 1.05
    if dv_m2sl == 0:
        w1_0 = WEIGHTS[1] * 0.65
        w1_1 = WEIGHTS[1] - w1_0
        d10n = st.ut_cr
    elif abs(dv_m2sl) > 2:
        w1_0 = WEIGHTS[1] * 0.0
        w1_1 = WEIGHTS[1] - w1_0
        d10n = 0
    else:
        w1_0 = WEIGHTS[1] * 0.5
        w1_1 = WEIGHTS[1] - w1_0
        if dv_m2sl > 0:
            d10n = st.ut_ct
        else:
            d10n = st.ut_cf
    score += log_score(w1_0, 6, d10n)
    sumn = st.ut_cr + st.ut_ct + st.ut_cf
    score += log_score(w1_1, 2.6, sumn)
    ds2c = distance_of(cur_idx, st.index, gll_len)
    score += lin_score(WEIGHTS[2], gll_len, ds2c)
    if ltu_idx == -1: score += WEIGHTS[3]
    else:
        ds2l = distance_of(ltu_idx, st.index, gll_len)
        if ds2l > 0:
            score += lin_score(WEIGHTS[3], gll_len, ds2l)
    return score

def find_len_closest(from_d: "dict[int, SameLenCs]", bias_l: int):
    closest: list[SameLenCs] = []
    min_diff = float('inf')
    for s_len, slc in from_d.items():
        diff = abs(s_len - bias_l)
        if diff > min_diff:
            continue
        elif diff < min_diff:
            min_diff = diff
            closest = [slc]
        else:
            closest.append(slc)
        if diff == 0: break
    return closest

def rep_align_use_best(cur_i: int, s_len: int, cpoem: ComposPoem):
    best_sws = (Sentence(cpoem.l_glt[cur_i], -1), float('-inf'))
    for slc in find_len_closest(cpoem.m_l2s, s_len):
        dv_m2sl = slc.c_len - s_len
        for stc in slc.stc_l:
            score = multi_dimen_score(stc,
                dv_m2sl, cpoem.t_len, cur_i, slc.ltu_i)
            if score > best_sws[1]: best_sws = (stc, score)
    best_stc = best_sws[0]
    bstc_len = len(best_stc.ltext)
    from_slc = cpoem.m_l2s[bstc_len]
    from_slc.ltu_i = best_stc.index
    diff = bstc_len - s_len
    if diff == 0:
        best_stc.ut_cr += 1
        return best_stc.ltext
    elif diff > 0:
        if diff > 3 and s_len < 4:
            return create_newl(cur_i, s_len, cpoem.l_glt)
        best_stc.ut_ct += 1
        klc = bstc_len > 13
        return trim_cutdown(best_stc.ltext, diff, klc)
    else:
        best_stc.ut_cf += 1
        if diff < -3:
            a = s_len // bstc_len
            b = -s_len % bstc_len
            best_stc.ut_cr += a
            btext = best_stc.ltext
            return btext*a + btext[b:]
        return fill_inflate(cur_i, -diff, cpoem.l_glt)

def check_can_trim(slen: int, glen: int):
    diff = glen - slen
    return 0 < diff < 3 and glen < 13, diff

def check_can_fill(slen: int, glen: int):
    diff = slen - glen
    return 0 < diff < 3 and glen > 1, diff

def align_generated_at(cur_i: int, s_len: int, cpoem: ComposPoem):
    g_line = cpoem.l_glt[cur_i]
    g_len = len(g_line)
    if g_len == s_len: return g_line
    can_trim, diff = check_can_trim(s_len, g_len)
    if can_trim:
        return trim_cutdown(g_line, diff, kp_lc=True)
    can_fill, diff = check_can_fill(s_len, g_len)
    if can_fill:
        return fill_inflate(cur_i, diff, cpoem.l_glt)
    return rep_align_use_best(cur_i, s_len, cpoem)

def force_align(g_lrc: str, s_ccr: str) -> str:
    _ST = time.perf_counter()
    glines = [l for l in g_lrc.splitlines() if l.strip()]
    slines = [l for l in s_ccr.splitlines() if l.strip()]
    glen = len(glines)
    slen = len(slines)
    assert glen * slen > 0, "No valid lines found in input."
    glines = math.ceil(slen / glen) * glines
    d_l2s: dict[int, SameLenCs] = {}
    for i, gline in enumerate(glines):
        line = gline.strip()
        llen = len(line)
        slc = d_l2s.setdefault(llen, SameLenCs([], llen))
        slc.stc_l.append(Sentence(line, i))
    cps_poem = ComposPoem(glines, d_l2s, glen)
    _ET = time.perf_counter()
    _logger.info(f"TSC: {_ET-_ST:.2f}s, setup ComposPoem")
    aligned = []
    for i, sline in enumerate(slines):
        _ST = time.perf_counter()
        aligned.append(
            align_generated_at(i, len(sline), cps_poem))
        _ET = time.perf_counter()
        _logger.info(f"TSC: {_ET-_ST:.2f}s, align_generated_at({i})")
    return '\n'.join(aligned)

ppt_tpl_0shot = """
主题：%s
结构：%s
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
        aligned = force_align(lyric, ccr_ccd)
        return aligned, lyric
    except:
        _logger.exception(f"mid-{mid} !! Lyric force-alignment failed.")
        _logger.warning(f"mid-{mid} ! Returning raw generated lyric.")
        return lyric, lyric

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
    print(f'wrap-unreplaced-structure:\n{args.structure}')
    structure = args.structure.replace('\\n', '\n')
    print(f'wrap-replaced-structure:\n{structure}')
    al, og = compose_poem(args.identity, args.purport, structure)
    print(f'ORIGINAL-GENERATED:\n{og}\n\nFORCE-ALIGNED:\n{al}')