import csv
import re
from dataclasses import dataclass
from io import StringIO

_regid = r'(?i)^\[(al|ar|au|by|length|offset|re|ti|ve):.*?\]'
_regft = r'^\[\d+:\d+(.\d+)?\]'

def _only_chs(line: str) -> bool:
    for c in line:
        if c.isspace():
            continue
        if not ('\u4e00' <= c <= '\u9fff'):
            return False
    return True

def _unify_sn(orign: str) -> str:
    nsname = orign.upper()
    match nsname:
        case '前' | '前奏' | 'PRELUDE':
            nsname = 'INTRO'
        case '主' | '主歌':
            nsname = 'VERSE'
        case '导' | '导歌':
            nsname = 'PRE-CHORUS'
        case '副' | '复' | '副歌' | '复歌':
            nsname = 'CHORUS'
        case '间' | '间奏':
            nsname = 'INTERLUDE'
        case '桥' | '桥段':
            nsname = 'BRIDGE'
        case '尾' | '尾奏' | '尾声' | 'ENDING':
            nsname = 'OUTRO'
    return nsname

def _csv_np(headrow: list[str], dlm: str = '|', ltm: str = '\n'):
    fsio = StringIO()
    csvw = csv.writer(fsio, delimiter=dlm, lineterminator=ltm)
    csvw.writerow(headrow)
    return (fsio, csvw)

class MsMuscle:
    _regcs = r'(?i)版权|授权|copy\s?right|未经.*[许同].*[不禁]|'\
             r'网易|云音乐|qq音|酷狗|咪咕音|酷我音|全民k歌|虾米|'\
             r'主题曲|审核|有限公司'
    _regnl = r'^.*[:：\(（][\W_]*'
    _regsg = r'(?i)^[\W_]*(m(?:ale)?|f(?:emale)?|d(?:uet)?|[男女]声?|合唱?)'\
             r'[\W_]*[1-4一二三四]?\s*(?:$|[^\w\s]+)'
    _regls = r'(?i)^[\W_]*(intro|prelude|verse|pre-chorus|chorus|'\
             r'interlude|bridge|outro|[前主副复导间桥尾][奏歌段声]?)[\W_]*'\
             r'[1-4一二三四]?\s*(?:$|[^\w\s]+)'
    _regss = r'说\s*[:：]\W*'

    @dataclass
    class RsoPs:
        cprstate: str | re.Pattern
        "copyright statement, for deletion with suspects"
        nonlyric: str | re.Pattern
        "non-lyric start reg, for deletion with suspects"
        gendermk: str | re.Pattern
        ""
        structmk: str | re.Pattern
        ""
        sepecial: str | re.Pattern
        ""
        lrcidtag: str | re.Pattern = _regid
        "for LRC ID tags, will be whole line deleted if match"
        lrcfttag: str | re.Pattern = _regft
        "LRC time tag that will be removed from line if match"

    _rsops = RsoPs(
        cprstate = re.compile(_regcs),
        nonlyric = re.compile(_regnl),
        gendermk = re.compile(_regsg),
        structmk = re.compile(_regls),
        sepecial = re.compile(_regss),
        lrcidtag = re.compile(_regid),
        lrcfttag = re.compile(_regft),
    )

    ClnTuple = tuple[int, str, dict[int, list[str]]|None, str|None]

    @staticmethod
    def cleaning(line: str,                 rule: RsoPs = _rsops,                 rpnw: bool = True,                 cnzh: bool = True) -> ClnTuple:
        if line is None or rule is None:
            raise ValueError('`line` or `rule` is None!')
        purelrct = line.strip()
        if ('\n' in purelrct) or ('\r' in purelrct):
            raise ValueError('`line` contains line break inside!')
        cotypeid = 0
        lrcfttag = ''
        if not purelrct:
            if purelrct != line:
                cotypeid = 3
            return (cotypeid, lrcfttag, None, purelrct)
        if idmatch := re.match(rule.lrcidtag, purelrct):
            return (1, idmatch.group(0), None, None)
        if tfmatch := re.search(rule.lrcfttag, line):
            lrcfttag = tfmatch.group(0)
            purelrct = line[tfmatch.end():].lstrip()
        if cnzh:
            purelrct = zhcvt(purelrct, 'zh-cn')
        suspects = {}
        if lsm := re.match(rule.structmk, purelrct):
            suspects[3] = [lsm.group(0), lsm.group(1)]
            purelrct = purelrct[lsm.end():].lstrip()
            cotypeid = 4
        if sgm := re.match(rule.gendermk, purelrct):
            suspects[2] = [sgm.group(0), sgm.group(1)]
            purelrct = purelrct[sgm.end():].lstrip()
            cotypeid = 4
        if nlmatch := re.match(rule.nonlyric, purelrct):
            suspect = nlmatch.group(0)
            if sm := re.search(rule.sepecial, suspect):
                suspects[4] = [suspect, sm.group(0)]
                cotypeid = 4
            else:
                suspects[0] = [suspect]
                return (2, lrcfttag, suspects, None)
        elif cm := re.findall(rule.cprstate, purelrct):
            suspects[1] = cm
            return (2, lrcfttag, suspects, None)
        if rpnw:
            purelrct = re.sub(r'[\W_]+', ' ', purelrct).strip()
        else:
            purelrct = re.sub(r'\s{2,}', ' ', purelrct).strip()
        if line != purelrct and 0 == cotypeid:
            cotypeid = 3
        suspects = suspects if suspects else None
        return (cotypeid, lrcfttag, suspects, purelrct)

    def __init__(self, /, creps: RsoPs = _rsops):
        self.creps = creps
        self.ocsvh = ['inputline', 'cleantype', 'suspects', 'cleanline', 'lineouput', 'reviewed', 'lineouput1', 'reviewed1']

    def wash_lrc_4llm(self, dirtyoriginal: str) -> tuple[str, str]:
        csvc, csvw = _csv_np(self.ocsvh[:6])
        returnc = ''
        for il in dirtyoriginal.splitlines():
            ctc, _, sus, pcl = MsMuscle.cleaning(il, self.creps)
            if pcl:
                returnc += (pcl + '\n')
            csvw.writerow([il, ctc, sus, pcl, pcl, 0])
        return (returnc.rstrip(), csvc.getvalue())

    def wash_lrc_4mfa(self, dirtyoriginal: str) -> tuple[str, str]:
        csvc, csvw = _csv_np(self.ocsvh[:6])
        csvrows = []
        mfaPcsr = -1
        returnc = ''
        for il in dirtyoriginal.splitlines():
            ctc, iot, sus, pcl = MsMuscle.cleaning(il, self.creps)
            cmfatlo = f'{iot+pcl}' if iot and pcl and _only_chs(pcl) else ''
            csvrows.append([il, ctc, sus, pcl, cmfatlo, 0])
            if -1 < mfaPcsr < len(csvrows):
                pmfarec = csvrows[mfaPcsr]
                pmfatlo = str(pmfarec[-2])
                if ',' not in pmfatlo and iot and ctc > 1:
                    pmfarec[-2] = pmfatlo.replace(']', f',{iot[1:]}', 1)
                    returnc += (pmfarec[-2] + '\n')
            if cmfatlo:
                mfaPcsr = len(csvrows) - 1
        if -1 < mfaPcsr < len(csvrows):
            pmfarec = csvrows[mfaPcsr]
            pmfatlo = str(pmfarec[-2])
            if ',' not in pmfatlo:
                pmfarec[-2] = pmfatlo.replace(']', ',-1]', 1)
                returnc += (pmfarec[-2] + '\n')
        csvw.writerows(csvrows)
        return (returnc.rstrip(), csvc.getvalue())

    W3pTuple = tuple[dict[str, list[str]], str, str]

    def wash_lrc_4pdt(self, llmgenerated: str) -> W3pTuple:
        csvc, csvw = _csv_np(self.ocsvh[:5])
        fclrcop = ''
        osldict: dict[str, list[str]] = {}
        cssdict: dict[str, int] = {}
        for il in llmgenerated.splitlines():
            ctc, _, sus, pcl = MsMuscle.cleaning(il, self.creps)
            lo = ''
            if ctc == 4 and sus and (lsm := sus.get(3)):
                if nsname := _unify_sn(lsm[1]):
                    css = cssdict.get(nsname, 0) + 1
                    cssdict[nsname] = css
                    lsncuri = f'{nsname} {css}'
                    osldict.setdefault(lsncuri, [])
                    lo = f'({lsncuri})\n'
            if pcl:
                lo += pcl + '\n'
                if osldict:
                    lastitem = list(osldict.items())[-1]
                    lastitem[1].append(pcl)
            fclrcop += lo
            lo = lo.rstrip().replace('\n', '\\n')
            csvw.writerow([il, ctc, sus, pcl, lo])
        return (osldict, fclrcop.rstrip(), csvc.getvalue())

    def wash_lrc_4ams(self, dirtyoriginal: str) -> str:
        returnc = ''
        for il in dirtyoriginal.splitlines():
            _, iot, _, pcl = MsMuscle.cleaning(il, self.creps, False, False)
            if pcl:
                returnc += (iot + pcl + '\n')
        return returnc.rstrip()

    def wash_lrc_4pap(self, dirtyoriginal: str) -> str:
        returnc = ''
        for il in dirtyoriginal.splitlines():
            _, iot, _, pcl = MsMuscle.cleaning(il, self.creps)
            if pcl:
                returnc += (iot + pcl + '\n')
        return returnc.rstrip()

    def wash_lrc_wosl(self, dirtyoriginal: str) -> str:
        returnc = ''
        for il in dirtyoriginal.splitlines():
            _, _, _, pcl = MsMuscle.cleaning(il, self.creps)
            if pcl:
                returnc += pcl.replace(' ', '')
        return returnc

def zhcvt(text, target):
    return text