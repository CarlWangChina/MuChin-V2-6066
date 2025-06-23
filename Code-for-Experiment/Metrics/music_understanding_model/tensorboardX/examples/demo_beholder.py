from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import Code_for_Experiment.Metrics.music_understanding_model.tensorboardX.tests.test_numpy
import Code_for_Experiment.Metrics.music_understanding_model.tensorboardX.tests.test_beholder
import time
from collections import namedtuple
import numpy as np
import beholder_lib

LOG_DIRECTORY = '/tmp/beholder-demo'
tensor_and_name = namedtuple('tensor_and_name', ['tensor', 'name'])

def beholder_pytorch():
    for i in range(1000):
        fake_param = [tensor_and_name(np.random.randn(128, 768, 3), 'test' + str(j)) for j in range(5)]
        arrays = [tensor_and_name(np.random.randn(128, 768, 3), 'test' + str(j)) for j in range(5)]
        beholder = beholder_lib.Beholder(logdir=LOG_DIRECTORY)
        beholder.update(
            trainable=fake_param,
            arrays=arrays,
            frame=np.random.randn(128, 128),
        )
        time.sleep(0.1)
        print(i)

if __name__ == '__main__':
    import os
    if not os.path.exists(LOG_DIRECTORY):
        os.makedirs(LOG_DIRECTORY)
    print(LOG_DIRECTORY)
    beholder_pytorch()