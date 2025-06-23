"""Generated LRC Objective Evaluation."""
import re
import jieba
from pypinyin import Style, pinyin
from pypinyin_dict.phrase_pinyin_data import large_pinyin
large_pinyin.load()
from pypinyin_dict.pinyin_data import cc_cedict
cc_cedict.load()

RHYTHM_TABLE = {
    'ㄚ': '01M',
    'ㄛ': '02B',
    'ㄜ': '03G',
    'ㄝ': '04J',
    'ㄓ': '05Z',
    'ㄔ': '05Z',
    'ㄕ': '05Z',
    'ㄖ': '05Z',
    'ㄗ': '05Z',
    'ㄘ': '05Z',
    'ㄙ': '05Z',
    'ㄦ': '06E',
    'ㄧ': '07Q',
    'ㄟ': '08W',
    'ㄞ': '09K',
    'ㄨ': '10G',
    'ㄩ': '11Y',
    'ㄡ': '12H',
    'ㄠ': '13H',
    'ㄢ': '14H',
    'ㄣ': '15H',
    'ㄤ': '16T',
    'ㄥ': '17G',
    'ㄧㄥ': '17G',
    'ㄨㄥ': '17G',
    '+ㄨㄥ': '18D',
    'ㄩㄥ': '18D',
}

def mapping_rhythm_name(w_zhuyin: str):
    zy_notone = re.sub(r'[˙ˊˇˋ]', '', w_zhuyin)
    rhythm = zy_notone[-1:]
    rtname = RHYTHM_TABLE.get(rhythm)
    if not rtname:
        return rhythm
    if rhythm != 'ㄥ':
        return rtname
    rhythm = zy_notone[-2:]
    if 'ㄩㄥ' == rhythm:
        return RHYTHM_TABLE.get(rhythm, '18D')
    elif 'ㄧㄥ' == rhythm:
        return RHYTHM_TABLE.get(rhythm, '17G')
    elif 'ㄨㄥ' == rhythm:
        return RHYTHM_TABLE.get(rhythm, '17G') if rhythm == zy_notone else RHYTHM_TABLE.get('+' + rhythm, '18D')
    else:
        return RHYTHM_TABLE.get(rhythm[-1:], '17G')

def find_last_zh_word(line: str):
    idx = len(line) - 1
    c = line[idx:]
    while (c < '\u4e00' or '\u9fff' < c) and idx > -1:
        idx -= 1
        c = line[idx:idx+1]
    return c, idx

def find_indices(pattern: str, text: str):
    matches = re.finditer(pattern, text)
    return [(m.start(), m.end()) for m in matches]

def split_repl_zhc2cr(sect_lrc: str, keepends=True):
    last_zhw_rinfo = {}
    rt_lines:list[str] = []
    og_lines = sect_lrc.splitlines(keepends)
    for i, line in enumerate(og_lines):
        _, last_zhw_idx = find_last_zh_word(line)
        phrases = list(jieba.cut(line[:last_zhw_idx+1]))
        zyl = pinyin(phrases, style=Style.BOPOMOFO, strict=False)
        if len(zyl) > 0 and len(zyl[-1]) > 0:
            lc_zhuyin = zyl[-1][0]
        else:
            lc_zhuyin = ''
        rhythm_name = mapping_rhythm_name(lc_zhuyin)
        last_zhw_rinfo.setdefault(rhythm_name, [])
        last_zhw_rinfo[rhythm_name].append((i, last_zhw_idx,))
        rt_lines.append(re.sub(r'[\u4e00-\u9fff]', 'c', line))
    for rn, infos in last_zhw_rinfo.items():
        if not rn or len(infos) < 2:
            continue
        for info in infos:
            il = info[0]
            ci = info[1]
            tl = rt_lines[il]
            rt_lines[il] = tl[:ci] + 'R' + tl[ci+1:]
    return rt_lines, og_lines

_RP_MSNT = r'\([\w-]+\)\n?'

def cnv_lrc_2cmp_infos(llm_gen: str):
    msnt_indices = find_indices(_RP_MSNT, llm_gen)
    mindices_len = len(msnt_indices)
    rt_alrc = ''
    rt_sidl = []
    rt_sncs = ''
    rt_slcs = ''
    for i, ct in enumerate(msnt_indices):
        if 0 == i and ct[0] > 0:
            rt_alrc += llm_gen[:ct[0]]
        msntl = llm_gen[ct[0]:ct[1]]
        sect = {'msntl': msntl}
        rt_alrc += msntl
        rt_sncs += msntl[1].upper()
        if (i+1) < mindices_len:
            nt = msnt_indices[i+1]
            sect_lrc = llm_gen[ct[1]:nt[0]]
        else:
            sect_lrc = llm_gen[ct[1]:]
        abs_sll, o_lines = split_repl_zhc2cr(sect_lrc)
        sect['lines'] = abs_sll
        sect['og_ls'] = o_lines
        abs_slrc = ''.join(abs_sll)
        rt_alrc += abs_slrc
        rt_sidl.append(sect)
        rt_slcs += chr(len(abs_sll)+48)
    return rt_alrc, rt_sidl, rt_sncs, rt_slcs

def cnv_mss_2cmp_infos(msstr: str):
    msnt_indices = find_indices(_RP_MSNT, msstr)
    mindices_len = len(msnt_indices)
    rt_ttlc = 0
    rt_sidl = []
    rt_sncs = ''
    rt_slcs = ''
    for i, ct in enumerate(msnt_indices):
        msntl = msstr[ct[0]:ct[1]]
        sect = {'msntl': msntl}
        rt_sncs += msntl[1].upper()
        if (i+1) < mindices_len:
            nt = msnt_indices[i+1]
            sp = msstr[ct[1]:nt[0]]
        else:
            sp = msstr[ct[1]:]
        s_lines = sp.splitlines(True)
        sect['lines'] = s_lines
        l_count = len(s_lines)
        rt_ttlc += l_count
        rt_sidl.append(sect)
        rt_slcs += chr(l_count+48)
    return rt_ttlc, rt_sidl, rt_sncs, rt_slcs

TOTAL_POINTS = 100
EXTRA_POINTS = 10
PH1_POINTS = TOTAL_POINTS * 0.10
PH2_POINTS = TOTAL_POINTS * 0.50
PH3_POINTS = TOTAL_POINTS * 0.20
PH4_POINTS = TOTAL_POINTS * 0.20
PH21_P_PCT = 0.65
PH22_P_PCT = 0.35

from difflib import HtmlDiff, SequenceMatcher, ndiff

def scoring(gen_lrc: str, o_msstr: str, am_p1sr=False):
    fnl_sum_score = 0.0
    detail_record = ''
    g_msstr, g_mssbl, g_snc_s, _ = cnv_lrc_2cmp_infos(gen_lrc)
    o_ttllc, o_mssbl, o_snc_s, _ = cnv_mss_2cmp_infos(o_msstr)
    detail_record += 'gen_lrc={!r}\no_msstr={!r}\n\ng_msstr={!r}\n\n'.format(gen_lrc, o_msstr, g_msstr)
    sm1 = SequenceMatcher(None, o_msstr, g_msstr)
    detail_record += "two abs-msstr sequences' overall comparison:\n"
    for tag, i1, i2, j1, j2 in sm1.get_opcodes():
        detail_record += '{:<9}o[{:<4}:{:>4}]  <->  g[{:<4}:{:>4}]\n'.format(tag, i1, i2, j1, j2)
    sm21 = SequenceMatcher(None, o_snc_s, g_snc_s)
    max_equ_slc_sum = 0
    matched2s_lcsum = 0.01
    wcm_amr = 1.0
    rc_ino = rc_ing = pll_rc = 0
    for m in sm21.get_matching_blocks():
        if 0 == m.size:
            break
        os_rg = range(m.a, m.a+m.size)
        gs_rg = range(m.b, m.b+m.size)
        for osi, gsi in zip(os_rg, gs_rg):
            olines = o_mssbl[osi]['lines']
            glines = g_mssbl[gsi]['lines']
            oslc = len(olines)
            gslc = len(glines)
            min_slc = min(oslc, gslc)
            for x in range(0, min_slc):
                olx = olines[x].strip()
                glx = glines[x].strip()
                olrc = olx.count('R')
                glcc = glx.count('c')
                glsc = glx.count(' ')
                glrc = glx.count('R')
                rc_ino += olrc
                rc_ing += glrc
                if glrc & olrc:
                    pll_rc += 1
                olxl = len(olx)
                min_lxcc = min(olxl, glcc+glsc+glrc)
                wcm_amr *= min_lxcc*2.0 / (olxl+len(glx))
            max_equ_slc_sum += min_slc
            matched2s_lcsum += oslc + gslc
    detail_record += "\nsection name code sequences' difference:\n"
    detail_record += ''.join(list(ndiff([o_snc_s+'\n'], [g_snc_s+'\n'], charjunk=None)))
    p1ws_sr = sm1.ratio()
    ph1score = PH1_POINTS * p1ws_sr
    fnl_sum_score += ph1score
    detail_record += '\nphase1   score:{:>8.4f}, p1ws_sr={:.8f}'.format(ph1score, p1ws_sr)
    acmp_sr = (p1ws_sr if am_p1sr else 1.0) * sm21.ratio()
    p21score = PH2_POINTS * PH21_P_PCT * acmp_sr
    fnl_sum_score += p21score
    detail_record += '\nphase2.1 score:{:>8.4f}, acmp_sr={:.8f} (am_p1sr is {})'.format(p21score, acmp_sr, am_p1sr)
    acmp_sr *= max_equ_slc_sum * 2.0 / matched2s_lcsum
    p22score = PH2_POINTS * PH22_P_PCT * acmp_sr
    fnl_sum_score += p22score
    detail_record += '\nphase2.2 score:{:>8.4f}, acmp_sr={:.8f}'.format(p22score, acmp_sr)
    ph3score = PH3_POINTS * wcm_amr * acmp_sr
    fnl_sum_score += ph3score
    detail_record += '\nphase3   score:{:>8.4f}, wcm_amr={:.8f}'.format(ph3score, wcm_amr)
    if (rc_ino + rc_ing) == 0:
        ttrc_sr = 0.0
    else:
        ttrc_sr = float(2*min(rc_ino, rc_ing) / (rc_ino+rc_ing))
    ph4score = PH4_POINTS * ttrc_sr * acmp_sr
    fnl_sum_score += ph4score
    detail_record += '\nphase4   score:{:>8.4f}, ttrc_sr={:.8f}'.format(ph4score, ttrc_sr)
    r_ratio = 0 if max_equ_slc_sum == 0 else rc_ing/max_equ_slc_sum
    extscore = EXTRA_POINTS * acmp_sr
    if 0.6 <= r_ratio <= 0.8:
        extscore *= 1.0
    elif pll_rc == rc_ino == rc_ing and pll_rc > 0:
        extscore *= 0.7
    elif (r_delta := abs(r_ratio-0.7)) <= 0.3:
        extscore *= 0.4 * (1-r_delta)
    else:
        extscore *= 0.0
    fnl_sum_score += extscore
    detail_record += '\nextra    score:{:>8.4f}, r_ratio={:.8f} (pll_rc={})'.format(extscore, r_ratio, pll_rc)
    detail_record += f'\n*FNL-SCORE-SUM: {fnl_sum_score}'
    als_diff_html = HtmlDiff(charjunk=None, tabsize=4, wrapcolumn=100).make_file(fromlines=o_msstr.splitlines(keepends=True), tolines=g_msstr.splitlines(keepends=True), fromdesc='original mss', todesc='generated mss')
    return fnl_sum_score, detail_record, als_diff_html

if __name__ == '__main__':
    o_msstr = '(verse)\ncccccccc\nccccccR\ncccccccc\nccccccR\n(verse)\ncccccccccR\nccccccccc\ncccccccccR\nccccccccc\n(chorus)\nccccccc\nccccccccR\nccccR\nccccccccR\n(chorus)\nccccccR\nccccccccR\nccccR\nccccccccc\n(verse)\ncccccccc\nccccccR\ncccccccc\nccccccR\n(verse)\ncccccccccR\nccccccccc\ncccccccccR\nccccccccc\n(chorus)\nccccccc\nccccccccR\nccccR\nccccccccR\n(chorus)\nccccccR\nccccccccR\nccccR\nccccccccc(verse)\ncccccccc\nccccccR\ncccccccc\nccccccR\n(verse)\ncccccccccR\nccccccccc\ncccccccccR\nccccccccc'
    gen_lrc = '(verse)\n如果一切可以重新来过\n我会在相遇的街口\n把你紧握再也不放手\n不会再让你远走\n(verse)\n如果上天能再给我一次\n爱你的机会不会辜负\n也许这次机会会更加\n更加珍惜不再错过你\n(chorus)\n因为曾经爱过你\n我的人生才会有奇迹\n因为有奇迹\n才能每天都能见到你\n(chorus)\n因为曾经爱过你\n我的人生才会有奇迹\n因为有奇迹\n才能每一分都为你而活\n(others)\n如果一切可以重新来过\n我会在相遇的街口\n把你紧握再也不放手\n不会再让你远走\n(verse)\n如果上天能再给我一次\n爱你的机会不会辜负\n也许这次机会会更加\n更加珍惜不再错过你\n(chorus)\n因为曾经爱过你\n我的人生才会有奇迹\n因为有奇迹\n才能每天都能见到你\n(chorus)\n因为曾经爱过你\n我的人生才会有奇迹\n因为有奇迹\n才能每一分都为你而活\n(verse)\n如果一切可以重新来过\n我会在相遇的街口\n把你紧握再也不放手\n不会再让你远走\n(verse)\n如果上天能再给我一次\n爱你的机会不会辜负\n也许这次机会会更加\n更加珍惜不再错过你'
    _, log, html = scoring(gen_lrc, o_msstr)
    print(log)