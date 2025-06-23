import os
import Code_for_Experiment.Metrics.music_understanding_model.tensorboardX.tests.test_numpy
from PIL import Image
import chainer
import chainer.cuda
from chainer import Variable
import numpy as np

def out_generated_image(gen, dis, rows, cols, seed, dst, writer):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        n_images = rows * cols
        xp = gen.xp
        z = Variable(xp.asarray(gen.make_hidden(n_images)))
        with chainer.using_config('train', False):
            x = gen(z)
        writer.add_image('img', x, trainer.updater.iteration)
    return make_image