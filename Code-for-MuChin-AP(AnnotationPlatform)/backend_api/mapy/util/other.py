def en_to_session_map(v: str):
    if v == 'intro':
        return '1'
    elif v == 'prelude':
        return '2'
    elif v == 'verse':
        return '3'
    elif v == 'pre-chorus':
        return '4'
    elif v == 'chorus':
        return '5'
    elif v == 'interlude':
        return '6'
    elif v == 'bridge':
        return '7'
    elif v == 'ending':
        return '8'
    elif v == 'outro':
        return '9'
    elif v == 'other':
        return '0'

def from_session_map(v: str):
    if v == '1':
        return 'intro'
    elif v == '2':
        return 'prelude'
    elif v == '3':
        return 'verse'
    elif v == '4':
        return 'pre-chorus'
    elif v == '5':
        return 'chorus'
    elif v == '6':
        return 'interlude'
    elif v == '7':
        return 'bridge'
    elif v == '8':
        return 'ending'
    elif v == '9':
        return 'outro'
    elif v == '0':
        return 'other'

def from_session_map_cn(v: str):
    if v == '1':
        return '前奏'
    elif v == '2':
        return '前奏'
    elif v == '3':
        return '主歌'
    elif v == '4':
        return '预副'
    elif v == '5':
        return '副歌'
    elif v == '6':
        return '间奏'
    elif v == '7':
        return '桥段'
    elif v == '8':
        return '尾奏'
    elif v == '9':
        return '尾奏'
    else:
        return '未知'

def cn_to_session_map(v: str):
    if v == '前奏':
        return '1'
    elif v == '主歌':
        return '3'
    elif v == '预副':
        return '4'
    elif v == '副歌':
        return '5'
    elif v == '间奏':
        return '6'
    elif v == '桥段':
        return '7'
    elif v == '尾奏':
        return '8'
    else:
        return '0'