import unittest
from Code_for_Experiment.Targeted_Training.Lyric_Generation_LLMs.lyric_cleaning.mb_lyric.lrcleaner import MsMuscle

_tc_lines_iom = {
    'Too many solo nights': (0, '', None, 'Too many solo nights'),
    '有没有人告诉你': (0, '', None, '有没有人告诉你'),
    '[al:The 60\'s Hits]': (1, '[al:The 60\'s Hits]', None, None),
    '[Ar:Chubby Checker]': (1, '[Ar:Chubby Checker]', None, None),
    '[au: Kal Mann], 1961]': (1, '[au: Kal Mann]', None, None),
    '[lenGth: 2:23] 巴拉巴拉巴拉': (1, '[lenGth: 2:23]', None, None),
    '[by: 此LRC文件的创建者]': (1, '[by: 此LRC文件的创建者]', None, None),
    '[offset:+/- 300]': (1, '[offset:+/- 300]', None, None),
    '[Re:?????]': (1, '[Re:?????]', None, None),
    '[ti:歌词(歌曲)的标题]': (1, '[ti:歌词(歌曲)的标题]', None, None),
    '[VE:程序的版本] xxxxxxxxxxxx': (1, '[VE:程序的版本]', None, None),
    '[00:00.00] 作词 : 陈小霞': (2, '[00:00.00]', {0: ['作词 :']}, None),
    '[00:01.000]策划：关大洲': (2, '[00:01.000]', {0: ['策划：']}, None),
    '[00:02.00]曲 ：高剑': (2, '[00:02.00]', {0: ['曲 ：']}, None),
    '[00:03.00]混音:梨酣': (2, '[00:03.00]', {0: ['混音:']}, None),
    '[00:05.29]【未经许可 禁止翻唱】': (2, '[00:05.29]', {1: ['未经许可 禁']}, None),
    '[00:04.32]未经同意，不得使用': (2, '[00:04.32]', {1: ['未经同意，不']}, None),
    '[00:02.40]本作品已获版权方授权': (2, '[00:02.40]', {1: ['版权', '授权']}, None),
    '[00:05.22]Copyright 2019': (2, '[00:05.22]', {1: ['Copyright']}, None),
    '[00:28.37]sia星雅：': (2, '[00:28.37]', {0: ['sia星雅：']}, None),
    '[00:56.64]邢香子：叹月照玉筝弦上手': (2, '[00:56.64]', {0: ['邢香子：']}, None),
    '[00:18.37] 紅紅落葉,長埋  “塵土‘內 ': (3, '[00:18.37]', None, '红红落叶 长埋 尘土 内'),
    '[00:19.92]lose dad’ s gun': (3, '[00:19.92]', None, 'lose dad s gun'),
    '[03:26.37]合：天真蓝': (4, '[03:26.37]', {2: ['合：', '合']}, '天真蓝'),
    '[02:30.93]M1 > GO!GO': (4, '[02:30.93]', {2: ['M1 >', 'M']}, 'GO GO'),
    '[04:32.13]尾：': (4, '[04:32.13]', {3: ['尾：', '尾']}, ''),
    '<Chorus-1>hah': (4, '', {3: ['<Chorus-1>', 'Chorus']}, 'hah'),
    'verse2 小丽啊': (0, '', None, 'verse2 小丽啊'),
    '[02:13.02] -你说:“没事，': (4, '[02:13.02]', {4: ['-你说:“', '说:“']}, '你说 没事'),
}

_tc_inp_llm_gen = """balabalabalabala2balabala...(prelude) Mama, just killed a manPut a gun against his head, pulled my trigger, now he's deadMama, life had just begunBut now I've gone and thrown it all away[Verse+1]在你身边我感到不同的自由你的脾气火爆却散发着真实的美身材或许不完美但却是你的标志因为你的直爽和可爱我无法自拔[Chorus+1]小丽，你的大眼睛像星星一样闪耀每次看着你我都觉得心跳加快我想告诉你，我对你的爱从未停止请收下我真挚的心意，小丽INTERLUDE:Oh, mamma mia, mamma miaMamma mia, let me goBeelzebub has a devil put aside for me, for me, for me[Verse+2]你的笑容像花朵绽放在春天你的声音像天使轻轻地呢喃即使你有时候发脾气也无所谓因为我愿意守护你的每一个瞬间[Chorus+2]小丽，你的大眼睛是我永远的依靠每次看着，你我都觉得世界变得明亮我想告诉你，我对你的爱是真实的请接受我深深的情感，小丽尾奏：OohOoh, yeah, ooh, yeahNothing really matters, anyone can seeNothing really mattersNothing really matters to me"""

_tc_ast_lg_fclo = """balabalabalabala2balabala(PRELUDE)Mama just killed a manPut a gun against his head pulled my trigger now he s deadMama life had just begunBut now I ve gone and thrown it all away(VERSE1)在你身边我感到不同的自由你的脾气火爆却散发着真实的美身材或许不完美但却是你的标志因为你的直爽和可爱我无法自拔(CHORUS1)小丽 你的大眼睛像星星一样闪耀每次看着你我都觉得心跳加快我想告诉你 我对你的爱从未停止请收下我真挚的心意 小丽(INTERLUDE)Oh mamma mia mamma miaMamma mia let me goBeelzebub has a devil put aside for me for me for me(VERSE2)你的笑容像花朵绽放在春天你的声音像天使轻轻地呢喃即使你有时候发脾气也无所谓因为我愿意守护你的每一个瞬间(CHORUS2)小丽 你的大眼睛是我永远的依靠每次看着 你我都觉得世界变得明亮我想告诉你 我对你的爱是真实的请接受我深深的情感 小丽(OUTRO/ENDING)OohOoh yeah ooh yeahNothing really matters anyone can seeNothing really mattersNothing really matters to me"""

_tc_ast_lg_alig = """(VERSE1)在你身边我感到不同的自由你的脾气火爆却散发着真实的美身材或许不完美但却是你的标志因为你的直爽和可爱我无法自拔(CHORUS1)小丽 你的大眼睛像星星一样闪耀每次看着你我都觉得心跳加快我想告诉你 我对你的爱从未停止请收下我真挚的心意 小丽(VERSE2)你的笑容像花朵绽放在春天你的声音像天使轻轻地呢喃即使你有时候发脾气也无所谓因为我愿意守护你的每一个瞬间(CHORUS2)小丽 你的大眼睛是我永远的依靠每次看着 你我都觉得世界变得明亮我想告诉你 我对你的爱是真实的请接受我深深的情感 小丽"""

_tc_inp_4mfa = """[offset:+300][00:00.00] 作词 : 樊冲[00:23.49]我要，你在我身旁[00:31.81]我说：“你为我梳妆”[00:39.20]这夜的风儿吹无时间标记的歌词行[00:47.14]我在他乡 望着月亮[00:51.37]Chorus(1)[01:34.95]都怪这Guitar 弹得太凄凉 (contain english)[01:42.77]合：欧 我要唱着歌[01:46.48]默默把你想 にほんご 我的情郎[01:50.70]F:你在何方 眼看天亮[02:14.99]我要 사랑해 美丽的衣裳[02:24.56]定位制作人：刘洲[02:30.45]这夜色太紧张[02:34.09]出品 : 创际音乐[02:38.32]「版权所有，未经许可请勿使用！」[02:40.00]你在何方 眼看天亮"""

_tc_ast_4mfa = """[00:23.49,00:31.81]我要 你在我身旁[00:31.81,00:39.20]我说 你为我梳妆[00:39.20,00:47.14]这夜的风儿吹[00:47.14,00:51.37]我在他乡 望着月亮[01:42.77,01:46.48]欧 我要唱着歌[01:50.70,02:14.99]你在何方 眼看天亮[02:30.45,02:34.09]这夜色太紧张[02:40.00,-1]你在何方 眼看天亮"""

class TestMsMuscle(unittest.TestCase):
    def setUp(self):
        self.msmuscle = MsMuscle()

    def test_cleaning(self):
        for line, expect in _tc_lines_iom.items():
            self.assertEqual(MsMuscle.cleaning(line), expect)

    def test_wash_lrc_4pdt(self):
        alig, fclo, _ = self.msmuscle.wash_lrc_4pdt(_tc_inp_llm_gen)
        self.assertEqual(fclo, _tc_ast_lg_fclo)
        self.assertEqual(alig, _tc_ast_lg_alig)

    def test_wash_lrc_4mfa(self):
        result = self.msmuscle.wash_lrc_4mfa(_tc_inp_4mfa)
        self.assertEqual(result, _tc_ast_4mfa)

if __name__ == '__main__':
    unittest.main()