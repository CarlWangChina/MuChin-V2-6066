from tensorboardX import SummaryWriter
import unittest

class SummaryWriterTest(unittest.TestCase):
    def test_summary_writer_ctx(self):
        with SummaryWriter(filename_suffix='.test') as writer:
            writer.add_scalar('test', 1)
        assert writer.file_writer is None

    def test_summary_writer_backcompat(self):
        with SummaryWriter(log_dir='/tmp/tbxtest') as writer:
            writer.add_scalar('test', 1)

    def test_summary_writer_close(self):
        passed = True
        try:
            writer = SummaryWriter()
            writer.close()
        except OSError:
            passed = False
        assert passed

    def test_windowsPath(self):
        dummyPath = "C:\\Downloads\\fjoweifj02utj43tj430"
        with SummaryWriter(dummyPath) as writer:
            writer.add_scalar('test', 1)
        import shutil
        shutil.rmtree(dummyPath)

    def test_pathlib(self):
        import sys
        if sys.version_info.major == 2:
            import pathlib2 as pathlib
        else:
            import pathlib
        p = pathlib.Path('./pathlibtest')
        with SummaryWriter(p) as writer:
            writer.add_scalar('test', 1)
        import shutil
        shutil.rmtree(str(p))