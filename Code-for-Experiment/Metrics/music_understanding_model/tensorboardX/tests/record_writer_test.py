from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import os
from Code_for_Experiment.Metrics.music_understanding_model.tensorboardX.tensorboardX.record_writer import RecordWriter
from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import PyRecordReader_New
import unittest

class RecordWriterTest(unittest.TestCase):
    def get_temp_dir(self):
        import tempfile
        return tempfile.mkdtemp()

    def test_expect_bytes_written(self):
        filename = os.path.join(self.get_temp_dir(), "expect_bytes_written")
        byte_len = 64
        w = RecordWriter(filename)
        bytes_to_write = b"x" * byte_len
        w.write(bytes_to_write)
        w.close()
        with open(filename, 'rb') as f:
            self.assertEqual(len(f.read()), (8 + 4 + byte_len + 4))

    def test_empty_record(self):
        filename = os.path.join(self.get_temp_dir(), "empty_record")
        w = RecordWriter(filename)
        bytes_to_write = b""
        w.write(bytes_to_write)
        w.close()
        r = PyRecordReader_New(filename)
        r.GetNext()
        self.assertEqual(r.record(), bytes_to_write)

    def test_record_writer_roundtrip(self):
        filename = os.path.join(self.get_temp_dir(), "record_writer_roundtrip")
        w = RecordWriter(filename)
        bytes_to_write = b"hello world"
        times_to_test = 50
        for _ in range(times_to_test):
            w.write(bytes_to_write)
        w.close()
        r = PyRecordReader_New(filename)
        for i in range(times_to_test):
            r.GetNext()
            self.assertEqual(r.record(), bytes_to_write)

if __name__ == '__main__':
    unittest.main()