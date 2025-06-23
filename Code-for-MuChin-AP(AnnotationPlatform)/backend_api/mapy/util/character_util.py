import re
from pypinyin import pinyin, Style
from pypinyin.style.bopomofo import BopomofoConverter

def replace_punctuation_with_space(text):
    pattern = r'[^\w\s]'
    return re.sub(pattern, ' ', text)

def replace_multiple_spaces_with_single_space(text):
    cleaned_text = re.sub(r' +', ' ', text)
    return cleaned_text

converter = BopomofoConverter()

def check_rhyme_by_section(section: []):
    current_rhyme_map = {}
    last_char_list = []
    for idx, value in enumerate(section):
        words = value['words']
        last_char = words[len(words) - 1]
        rhyme = get_rhyme_from_char(last_char)
        last_char_list.append(last_char)
        for i in range(0, len(rhyme)):
            new_str = rhyme[i:len(rhyme)]
            if new_str in rhyme_map:
                trans = rhyme_trans_map[new_str]
                if trans in current_rhyme_map:
                    v = current_rhyme_map[trans]
                    v.append(idx)
                    current_rhyme_map[trans] = v
                else:
                    current_rhyme_map[trans] = [idx]
                break
    for key in current_rhyme_map:
        value = current_rhyme_map[key]
        if len(value) <= 1:
            section[value[0]]['pinyin'] = pinyin(last_char_list[value[0]], style=Style.NORMAL)[0][0]
            section[value[0]]['rhyme'] = ""
            continue
        for idx in value:
            section[idx]['pinyin'] = pinyin(last_char_list[idx], style=Style.NORMAL)[0][0]
            section[idx]['rhyme'] = key
    return section

def check_rhyme_by_piny(section: []):
    current_rhyme_map = {}
    for idx, value in enumerate(section):
        rhyme = get_rhyme_from_pinyin(value['pinyin'])
        if rhyme is None:
            continue
        for i in range(0, len(rhyme)):
            new_str = rhyme[i:len(rhyme)]
            if new_str in rhyme_map:
                trans = rhyme_trans_map[new_str]
                if trans in current_rhyme_map:
                    v = current_rhyme_map[trans]
                    v.append(idx)
                    current_rhyme_map[trans] = v
                else:
                    current_rhyme_map[trans] = [idx]
                break
    for key in current_rhyme_map:
        value = current_rhyme_map[key]
        if len(value) <= 1:
            continue
        for idx in value:
            section[idx]['rhyme'] = key
    return section

def get_pin_form_char(char: str):
    return pinyin(char, style=Style.NORMAL)[0][0]

def get_rhyme_from_char(char: str):
    piny = pinyin(char, style=Style.NORMAL)[0][0]
    return get_rhyme_from_pinyin(piny)

def get_rhyme_from_pinyin(piny: str):
    if piny == '':
        return
    value = converter.to_bopomofo(pinyin=piny)
    new = re.sub(r'[˙ˊˇˋ]', '', value)
    if new in rhyme_map:
        result = new
    else:
        first = converter.to_bopomofo_first(pinyin=piny)
        result = new.replace(first, '')
    if result == '':
        return '⑤'
    elif result == 'ㄨㄥ' and piny.__contains__('ong'):
        return '-' + result
    else:
        return result

def create_rhyme_result_list_by_position(lrc_list, cur_section, start, end):
    sub_lrc_list = []
    if len(cur_section) == 2:
        section = from_session_map_cn(cur_section[0]) + str(ord(cur_section[-1:]) - 96)
    else:
        section = ""
    for i in range(start, end + 1):
        sub_lrc_list.append({
            'line': i,
            'type': section,
            'words': lrc_list[i],
        })
    return check_rhyme_by_section(sub_lrc_list)

def calcul_rhyme_by_pinyin_or_char(data: list):
    format_data = []
    for i in data:
        pyin = i['pinyin'].strip()
        if pyin != '':
            format_data.append({
                'pinyin': pyin
            })
        else:
            words = i['words'].strip()
            format_data.append({
                'pinyin': get_pin_form_char(words[-1:])
            })
    return check_rhyme_by_piny(format_data)