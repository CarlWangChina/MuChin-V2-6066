import unittest
from glrc_obj_eval import *

class TestCkptObjEval(unittest.TestCase):
    def test_find_last_zh_word(self):
        inp_lines = [
            '',
            "There's not a single Chinese(zh-cn) character!",
            '好了现在有啦~\n',
            '我们期望的 正常的歌词\n',
        ]
        o_expects = [
            ('', -1),
            ('', -1),
            ('啦', 5),
            ('词', 10),
        ]
        for i, l in enumerate(inp_lines):
            r = find_last_zh_word(l)
            self.assertEqual(r, o_expects[i])

    @unittest.skip(reason="very basic re usage and list comprehension")
    def test_find_indices(self):
        pass

    def test_split_repl_zhc2cr(self):
        inp_lrcs = [
            "\nGood morning!大家好\n中文最叼~\n",
            "他们说：“没有文化的人不伤心”\n他不会伤心\n他也会伤心\n",
            "爱过一次一次无休对峙\n恨过一次不懂感情的认知\n不难想象我的一味相思\n全是你一个人\n多情的放肆\n",
        ]
        o_expects = [
            ['\n', 'Good morning!ccR\n', 'cccR~\n'],
            ['ccc：“ccccccccR”\n', 'ccccR\n', 'ccccR\n'],
            ['cccccccccR\n', 'ccccccccccR\n', 'cccccccccR\n', 'cccccc\n', 'ccccR\n'],
        ]
        for i, lrcstr in enumerate(inp_lrcs):
            r, _ = split_repl_zhc2cr(lrcstr)
            self.assertEqual(r, o_expects[i])

if __name__ == '__main__':
    unittest.main()