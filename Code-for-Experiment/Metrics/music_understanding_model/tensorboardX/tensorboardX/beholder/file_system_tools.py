from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle

def write_file(contents, path, mode='wb'):
    with open(path, mode) as new_file:
        new_file.write(contents)

def write_pickle(obj, path):
    with open(path, 'wb') as new_file:
        pickle.dump(obj, new_file)

def read_pickle(path, default=None):
    try:
        with open(path, 'rb') as pickle_file:
            result = pickle.load(pickle_file)
    except (FileNotFoundError, EOFError):
        result = default
    return result