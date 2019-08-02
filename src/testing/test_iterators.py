from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf
import matplotlib.pyplot as plt

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config as config
from src.config import get_config, prepare_dirs, save_config
from src.data_loader import DataLoader
from src.RunModel import RunModel
from src.util.load_data import example_run
from src.trainer import HMRTrainer

import src.util.data_utils as du

def main(config):
    data_loader = DataLoader(config)
    dataset = data_loader.load()

    dataset_a = dataset.take(1000)
    dataset_b = dataset.skip(0)

#    dataset_a = tf.data.Dataset.range(10).shuffle(1000).repeat()
#   dataset_b = tf.data.Dataset.range(20, 25).shuffle(1000).repeat()


    iterator = dataset_a.make_initializable_iterator()

    iterator_init_op_a = iterator.make_initializer(dataset_a)
    iterator_init_op_b = iterator.make_initializer(dataset_b)

    image, seg, kp = iterator.get_next()

    with tf.train.MonitoredTrainingSession() as sess:
        sess.run(iterator_init_op_a)
        res = sess.run(image)
        print(res.shape)
        res = sess.run(image)
        print(res.shape)
        sess.run(iterator_init_op_b)

        count = 0
        while True:
            count = count + 1
            print(count)

            res = sess.run(image)


if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    main(config)
