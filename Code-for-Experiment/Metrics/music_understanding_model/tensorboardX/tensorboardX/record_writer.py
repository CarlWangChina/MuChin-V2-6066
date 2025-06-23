import copy
import io
import os.path
import re
import struct

try:
    import boto3
    S3_ENABLED = True
except ImportError:
    S3_ENABLED = False

from .crc32c import crc32c

_VALID_OP_NAME_START = re.compile('^[A-Za-z0-9.]')
_VALID_OP_NAME_PART = re.compile('[A-Za-z0-9_.\\-/]+')

REGISTERED_FACTORIES = {}

def register_writer_factory(prefix, factory):
    if ':' in prefix:
        raise ValueError('prefix cannot contain a :')
    REGISTERED_FACTORIES[prefix] = factory

def directory_check(path):
    try:
        prefix = path.split(':')[0]
        factory = REGISTERED_FACTORIES[prefix]
        return factory.directory_check(path)
    except KeyError:
        if not os.path.exists(path):
            os.makedirs(path)

def open_file(path):
    try:
        prefix = path.split(':')[0]
        factory = REGISTERED_FACTORIES[prefix]
        return factory.open(path)
    except KeyError:
        return open(path, 'wb')

class S3RecordWriter(object):
    def __init__(self, path):
        if not S3_ENABLED:
            raise ImportError("boto3 must be installed for S3 support.")
        self.path = path
        self.buffer = io.BytesIO()

    def __del__(self):
        self.close()

    def bucket_and_path(self):
        path = self.path
        if path.startswith("s3://"):
            path = path[len("s3://"):]
        bp = path.split("/")
        bucket = bp[0]
        path = path[1 + len(bucket):]
        return bucket, path

    def write(self, val):
        self.buffer.write(val)

    def flush(self):
        s3 = boto3.client('s3')
        bucket, path = self.bucket_and_path()
        upload_buffer = copy.copy(self.buffer)
        upload_buffer.seek(0)
        s3.upload_fileobj(upload_buffer, bucket, path)

    def close(self):
        self.flush()

class S3RecordWriterFactory(object):
    def open(self, path):
        return S3RecordWriter(path)

    def directory_check(self, path):
        pass

register_writer_factory("s3", S3RecordWriterFactory())

class RecordWriter(object):
    def __init__(self, path):
        self._name_to_tf_name = {}
        self._tf_names = set()
        self.path = path
        self._writer = None
        self._writer = open_file(path)

    def write(self, data):
        w = self._writer.write
        header = struct.pack('Q', len(data))
        w(header)
        w(struct.pack('I', masked_crc32c(header)))
        w(data)
        w(struct.pack('I', masked_crc32c(data)))

    def flush(self):
        self._writer.flush()

    def close(self):
        self._writer.close()

def masked_crc32c(data):
    x = u32(crc32c(data))
    return u32(((x >> 15) | u32(x << 17)) + 0xa282ead8)

def u32(x):
    return x & 0xffffffff

def make_valid_tf_name(name):
    if not _VALID_OP_NAME_START.match(name):
        name = '.' + name
    return '_'.join(_VALID_OP_NAME_PART.findall(name))