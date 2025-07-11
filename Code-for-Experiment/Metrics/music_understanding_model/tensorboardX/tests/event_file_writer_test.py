from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob
import os
from Code_for_Experiment.Metrics.music_understanding_model.tensorboardX.tensorboardX.event_file_writer import EventFileWriter
from Code_for_Experiment.Metrics.music_understanding_model.tensorboardX.tensorboardX.event_file_writer import EventFileWriter as _AsyncWriter
from tensorboardX.proto import event_pb2
from tensorboardX.proto.summary_pb2 import Summary
from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import PyRecordReader_New
import unittest

class EventFileWriterTest(unittest.TestCase):
    def get_temp_dir(self):
        import tempfile
        return tempfile.mkdtemp()

    def test_event_file_writer_roundtrip(self):
        _TAGNAME = 'dummy'
        _DUMMY_VALUE = 42
        logdir = self.get_temp_dir()
        w = EventFileWriter(logdir)
        summary = Summary(value=[Summary.Value(tag=_TAGNAME, simple_value=_DUMMY_VALUE)])
        fakeevent = event_pb2.Event(summary=summary)
        w.add_event(fakeevent)
        w.close()
        event_files = sorted(glob.glob(os.path.join(logdir, '*')))
        self.assertEqual(len(event_files), 1)
        r = PyRecordReader_New(event_files[0])
        r.GetNext()
        r.GetNext()
        self.assertEqual(fakeevent.SerializeToString(), r.record())

    def test_setting_filename_suffix_works(self):
        logdir = self.get_temp_dir()
        w = EventFileWriter(logdir, filename_suffix='.event_horizon')
        w.close()
        event_files = sorted(glob.glob(os.path.join(logdir, '*')))
        self.assertEqual(event_files[0].split('.')[-1], 'event_horizon')

    def test_async_writer_without_write(self):
        logdir = self.get_temp_dir()
        w = EventFileWriter(logdir)
        w.close()
        event_files = sorted(glob.glob(os.path.join(logdir, '*')))
        r = PyRecordReader_New(event_files[0])
        r.GetNext()
        s = event_pb2.Event.FromString(r.record())
        self.assertEqual(s.file_version, "brain.Event:2")

class AsyncWriterTest(unittest.TestCase):
    def get_temp_dir(self):
        import tempfile
        return tempfile.mkdtemp()

    def test_async_writer_write_once(self):
        foldername = os.path.join(self.get_temp_dir(), "async_writer_write_once")
        w = _AsyncWriter(foldername)
        filename = w._ev_writer._file_name
        bytes_to_write = b"hello world"
        w.add_event(bytes_to_write)
        w.close()
        with open(filename, 'rb') as f:
            self.assertEqual(f.read(), bytes_to_write)

    def test_async_writer_write_queue_full(self):
        filename = os.path.join(self.get_temp_dir(), "async_writer_write_queue_full")
        w = _AsyncWriter(filename)
        bytes_to_write = b"hello world"
        repeat = 100
        for i in range(repeat):
            w.write(bytes_to_write)
        w.close()
        with open(filename, 'rb') as f:
            self.assertEqual(f.read(), bytes_to_write * repeat)

    def test_async_writer_write_one_slot_queue(self):
        filename = os.path.join(self.get_temp_dir(), "async_writer_write_one_slot_queue")
        w = _AsyncWriter(filename, max_queue_size=1)
        bytes_to_write = b"hello world"
        repeat = 10
        for i in range(repeat):
            w.write(bytes_to_write)
        w.close()
        with open(filename, 'rb') as f:
            self.assertEqual(f.read(), bytes_to_write * repeat)

    def test_async_writer_close_triggers_flush(self):
        filename = os.path.join(self.get_temp_dir(), "async_writer_close_triggers_flush")
        w = _AsyncWriter(filename)
        bytes_to_write = b"x" * 64
        w.write(bytes_to_write)
        w.close()
        with open(filename, 'rb') as f:
            self.assertEqual(f.read(), bytes_to_write)

    def test_write_after_async_writer_closed(self):
        filename = os.path.join(self.get_temp_dir(), "write_after_async_writer_closed")
        w = _AsyncWriter(filename)
        bytes_to_write = b"x" * 64
        w.write(bytes_to_write)
        w.close()
        with self.assertRaises(IOError):
            w.write(bytes_to_write)
        with open(filename, 'rb') as f:
            self.assertEqual(f.read(), bytes_to_write)

if __name__ == '__main__':
    unittest.main()