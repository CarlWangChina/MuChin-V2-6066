import logging
import re
import openai
import mb_lyric.llmrelate as LLMr
from Code_for_Experiment.Targeted_Training.Lyric_Generation_LLMs.lyric_cleaning.mb_lyric.exception import LlmBadReturn
from Code_for_Experiment.Targeted_Training.Lyric_Generation_LLMs.lyric_cleaning.mb_lyric.lrcleaner import MsMuscle

class Freddie:
    _logger = logging.getLogger(__name__ + '.Freddie')

    def __init__(self, mname='gpt-4', gtemp=1.2, atemp=0.6, retry=3, maxchwdiff=1):
        self._lg = Freddie._logger
        if retry not in range(0, 6):
            raise ValueError('retry must be in [0, 6)!')
        self.modelname = mname
        self.genertemp = gtemp
        self.analytemp = atemp
        self.fmaxretry = retry
        self.cleaner = MsMuscle()
        self._relsv = re.compile(r'(?i)[\W_]verse')
        self._relsc = re.compile(r'(?i)[\W_]chorus')
        self._reldl = re.compile(r'(?i)[\W_]+DIVIDER[\W_]+')
        self._resnm = re.compile(r'(?i)\s+name\x20*=([^\r\n]*)\s+')
        self._revew = re.compile(r'(?i)\b(excited|delighted|happy|content|relaxed|clam|tired|bored|depressed|angry|frustrated|tense)\b')
        if not openai.api_key:
            from dotenv import find_dotenv, load_dotenv
            load_dotenv(find_dotenv(), override=True)
            import os
            openai.api_key = os.getenv("OPENAI_API_KEY")

    def _req_llm(self, /, mname, sysct, usrct, mtemp) -> str:
        response = {}
        self._lg.info(f'LLM raw response:\n{response}')
        return response['choices'][0]['message']['content']

    def _choose_best(self, lyrics: str) -> tuple[str, str]:
        origll = self._reldl.split(lyrics)
        songname = ''
        vcolyric = ''
        for i, lwn in enumerate(origll):
            lwon = lwn
            if m := self._resnm.search(lwn):
                songname = m.group(1).strip()
                lwon = lwn[:m.start()] + '\n' + lwn[m.end():]
            osldict = self.cleaner.wash_lrc_4pdt(lwon)[0]
            v1l = osldict.get('VERSE 1', [])
            c1l = osldict.get('CHORUS 1', [])
            if len(v1l) < 2 or len(c1l) < 2:
                self._lg.warning(f'skip {i+1}th lyric cause "VERSE" or "CHORUS" part line lenth < 2!')
                continue
            v2l = osldict.get('VERSE 2', [])
            if len(v2l) < 2:
                v2l = v1l
            vminlen = min(len(v1l), len(v2l))
            osldict['VERSE 1'] = v1l[:vminlen]
            osldict['VERSE 2'] = v2l[:vminlen]
            c2l = osldict.get('CHORUS 2', [])
            if len(c2l) < 2:
                c2l = c1l
            cminlen = min(len(c1l), len(c2l))
            osldict['CHORUS 1'] = c1l[:cminlen]
            osldict['CHORUS 2'] = c2l[:cminlen]
        if not songname:
            self._lg.warning('no song name found, use default!')
            songname = '未命名'
        return (songname, vcolyric)

    def gen_lyric(self, taskid: str, usrinput: str) -> tuple[str, str]:
        if not usrinput:
            raise ValueError('Must provide a non-blank `usrinput`!')
        self._lg.info(f'start generating lyric for the task {taskid} with user input: {usrinput}')
        curretry = 0
        while curretry <= self.fmaxretry:
            if curretry > 0:
                self._lg.info(f'start {curretry}th retry')
            respctt = ''
            try:
                respctt = self._req_llm(mname=self.modelname, sysct=LLMr.spt_lyric_gen, usrct=f'输入：{usrinput}', mtemp=self.genertemp)
                s = respctt.find('START')
                e = respctt.find('END')
                if s == -1 or e == -1:
                    raise LlmBadReturn('no "START" or "END" mark!')
                return self._choose_best(respctt[s+5:e])
            except Exception as e:
                self._lg.error(f'Error Occured', exc_info=True)
                if respctt:
                    self._lg.warning(f'*exc caused respctt record*, tid-{taskid} of {curretry+1}th lyric-gen:\n{respctt}')
            finally:
                curretry += 1
        raise RuntimeError('Still failed to generate a valid lyric after done {self.fmaxretry} retries!')

    def a_emotion(self, taskid: str, usrinput: str) -> str:
        if not (taskid and usrinput):
            raise ValueError('taskid and usrinput must be provided!')
        curretry = 0
        while curretry <= self.fmaxretry:
            respctt = ''
            try:
                respctt = self._req_llm(mname=self.modelname, sysct=LLMr.spt_rcme_anls, usrct=f'input: {usrinput}', mtemp=self.analytemp)
                if emotions := self._revew.findall(respctt):
                    return str(emotions[-1]).capitalize()
                else:
                    raise LlmBadReturn('no valid emotion returned!')
            except Exception as e:
                self._lg.error(f'Error Occured', exc_info=True)
                if respctt:
                    self._lg.warning(f'*exc caused respctt record*, tid-{taskid} of {curretry+1}th emotion-a:\n{respctt}')
            finally:
                curretry += 1
        raise RuntimeError(f'Still failed to get an valid emotion after done {self.fmaxretry} retries!')