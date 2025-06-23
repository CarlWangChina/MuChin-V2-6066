from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import socket
import threading
import time
import six
from .proto import event_pb2
from .record_writer import RecordWriter, directory_check

class EventsWriter(object):
    def __init__(self, file_prefix, filename_suffix=''):
        self._file_name = file_prefix + ".out.tfevents." + str(time.time())[:10] + "." + \
            socket.gethostname() + filename_suffix
        self._num_outstanding_events = 0
        self._py_recordio_writer = RecordWriter(self._file_name)
        self._event = event_pb2.Event()
        self._event.wall_time = time.time()
        self._event.file_version = 'brain.Event:2'
        self._lock = threading.Lock()
        self.write_event(self._event)

    def write_event(self, event):
        if not isinstance(event, event_pb2.Event):
            raise TypeError("Expected an event_pb2.Event proto, but got %s" % type(event))
        return self._write_serialized_event(event.SerializeToString())

    def _write_serialized_event(self, event_str):
        with self._lock:
            self._num_outstanding_events += 1
            self._py_recordio_writer.write(event_str)

    def flush(self):
        with self._lock:
            self._num_outstanding_events = 0
            self._py_recordio_writer.flush()
        return True

    def close(self):
        return_value = self.flush()
        with self._lock:
            self._py_recordio_writer.close()
        return return_value

class EventFileWriter(object):
    def __init__(self, logdir, max_queue_size=10, flush_secs=120, filename_suffix=''):
        self._logdir = logdir
        directory_check(self._logdir)
        self._event_queue = six.moves.queue.Queue(max_queue_size)
        self._ev_writer = EventsWriter(os.path.join(
            self._logdir, "events"), filename_suffix)
        self._flush_secs = flush_secs
        self._closed = False
        self._worker = _EventLoggerThread(self._event_queue, self._ev_writer,
                                          flush_secs)
        self._worker.start()

    def get_logdir(self):
        return self._logdir

    def reopen(self):
        if self._closed:
            self._closed = False
            self._worker = _EventLoggerThread(
                self._event_queue, self._ev_writer, self._flush_secs
            )
            self._worker.start()

    def add_event(self, event):
        if not self._closed:
            self._event_queue.put(event)

    def flush(self):
        if not self._closed:
            self._event_queue.join()
            self._ev_writer.flush()

    def close(self):
        if not self._closed:
            self.flush()
            self._worker.stop()
            self._ev_writer.close()
            self._closed = True

class _EventLoggerThread(threading.Thread):
    def __init__(self, queue, record_writer, flush_secs):
        threading.Thread.__init__(self)
        self.daemon = True
        self._queue = queue
        self._record_writer = record_writer
        self._flush_secs = flush_secs
        self._next_flush_time = 0
        self._has_pending_data = False
        self._shutdown_signal = object()

    def stop(self):
        self._queue.put(self._shutdown_signal)
        self.join()

    def run(self):
        while True:
            now = time.time()
            queue_wait_duration = self._next_flush_time - now
            data = None
            try:
                if queue_wait_duration > 0:
                    data = self._queue.get(True, queue_wait_duration)
                else:
                    data = self._queue.get(False)
                if data == self._shutdown_signal:
                    return
                self._record_writer.write_event(data)
                self._has_pending_data = True
            except six.moves.queue.Empty:
                pass
            finally:
                if data:
                    self._queue.task_done()
            now = time.time()
            if now > self._next_flush_time:
                if self._has_pending_data:
                    self._record_writer.flush()
                    self._has_pending_data = False
                self._next_flush_time = now + self._flush_secs