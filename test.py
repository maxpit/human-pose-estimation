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
    prepare_dirs(config)

    # Load data on CPU
    with tf.device("/cpu:0"):
        data_loader = DataLoader(config)
        dataset = data_loader.load()
        #smpl_loader = data_loader.get_smpl_loader()

#    iterator = dataset.make_one_shot_iterator()

#    next_element = iterator.get_next()
    
    # The op for initializing the variables.
#    init_op = tf.group(tf.global_variables_initializer(),
#                       tf.local_variables_initializer())
    
#    with tf.Session()  as sess:
#
#        sess.run(init_op)
#
#        coord = tf.train.Coordinator()
#        threads = tf.train.start_queue_runners(coord=coord)
#        
#        for _ in range(num_epochs):
#            while True:
#                try:
#                    img, seg_gt, label = sess.run(next_element)
#                    plt.figure(figsize=(8,8))
#
#                    #for i in range (config.batch_size):
#                    plt.subplot(2,2,1)
#                    plt.imshow(img[0])
#
#                    plt.subplot(2,2,2)
#                    plt.imshow(seg_gt[0].squeeze())
#                    plt.subplot(2,2,3)
#                    plt.imshow(img[1])
#
#                    plt.subplot(2,2,4)
#                    plt.imshow(seg_gt[1].squeeze())
#                        
#                    plt.show()
#                except tf.errors.OutOfRangeError:
#                    break
#
#        coord.request_stop()
#        coord.join(threads)
 
    trainer = HMRTrainer(config, dataset)
    save_config(config)
    trainer.train()
 
if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    main(config)
