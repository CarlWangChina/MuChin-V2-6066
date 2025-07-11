import logging
logging.basicConfig(
    format='%(asctime)s|%(levelname)s|%(process)d-%(thread)d|%(name)s: %(message)s',
    level=logging.INFO
)
_logger = logging.getLogger(__name__)
import re
import torch
from peft import AutoPeftModelForCausalLM as AutoModelForCausalLM
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers.generation import GenerationConfig

def init_all_models():
    global tknzer_libai, model_libai, lac, tknzer_bert, model_bert
    tknzer_libai = AutoTokenizer.from_pretrained(
        '/data/shared/Qwen/base-model',
        trust_remote_code=True,
        resume_download=True,
    )
    model_libai = AutoModelForCausalLM.from_pretrained(
        '/data/shared/Qwen/final-ckpt/',
        device_map='auto',
        trust_remote_code=True,
        resume_download=True,
    ).eval()
    model_libai.generation_config = GenerationConfig.from_pretrained(
        '/data/shared/Qwen/final-ckpt/',
        trust_remote_code=True,
        resume_download=True,
    )
    tknzer_bert = AutoTokenizer.from_pretrained(
        '/data/shared/Qwen/bert-bc'
    )
    model_bert = AutoModelForMaskedLM.from_pretrained(
        '/data/shared/Qwen/bert-bc'
    )

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

def assemble_mask(
    m_count: int,
    tail_ij: str,
    a_lines: "list[str]",
    g_lines: "list[str]",
    cur_idx: int,
    punc_mk: str='，',
):
    assert m_count > 0, "Invalid mask number."
    mask_l = '[MASK]'*m_count + tail_ij
    wc_sum = m_count + len(tail_ij)
    pl_idx = cur_idx - 1
    nl_idx = cur_idx + 1
    arange = range(len(a_lines))
    grange = range(len(g_lines))
    while m_count / wc_sum > 0.0625:
        if pl_idx in arange:
            pl = a_lines[pl_idx]
            mask_l = pl + punc_mk + mask_l
            wc_sum += len(pl)
            pl_idx -= 1
        elif nl_idx in grange:
            nl = g_lines[nl_idx]
            mask_l = mask_l + punc_mk + nl
            wc_sum += len(nl)
            nl_idx += 1
        else:
            break
    return mask_l

def ajust_eline_words(
    gl_line: str,
    cr_line: str,
    a_lines: "list[str]",
    g_lines: "list[str]",
    cur_idx: int,
):
    gl_len = len(gl_line)
    cl_len = len(cr_line)
    if gl_len == cl_len:
        return gl_line
    elif gl_len > cl_len:
        fill_c = cl_len - 1
        tjoint = gl_line[-1]
        if not fill_c:
            return tjoint
        mask_l = assemble_mask(fill_c, tjoint,
                                a_lines, g_lines, cur_idx)
    else:
        fill_c = cl_len - gl_len
        tjoint = gl_line
        mask_l = assemble_mask(fill_c, gl_line,
                                a_lines, g_lines, cur_idx)
    return predict_masked(mask_l) + tjoint

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
            ajust_eline_words(gl, sl, aligned, glines, i)
        )
    return '\n'.join(aligned)

ppt_tpl_0shot = "%s\n%s"

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
    except:
        _logger.exception(f"mid-{mid} !! Lyric force-alignment failed.")
        _logger.warning(f"mid-{mid} ! Returning raw generated lyric.")
        return lyric, lyric
    return aligned, lyric

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