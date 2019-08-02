from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config as config
from src.config import get_config, prepare_dirs, save_config
from src.data_loader import DataLoader
#from src.RunModel import RunModel
#from src.util.load_data import example_run
from src.models import Encoder_resnet, Critic_network
import deepdish as dd

from os.path import join, dirname
#import src.util.data_utils as du

def main(config):
    prepare_dirs(config)

    # Load data on CPU
    with tf.device("/cpu:0"):
        data_loader = DataLoader(config)
        dataset = data_loader.load()
        smpl_dataset = data_loader.get_smpl_loader()

#    iterator = dataset.make_one_shot_iterator()

#    trainer = HMRTrainer(config, dataset, smpl_loader)
#    save_config(config)
#    trainer.train()
    dataset = dataset#.shuffle(buffer_size=10000)
    dataset = dataset.batch(8)
#    smpl_dataset = smpl_dataset.shuffle(buffer_size=1000).repeat()
#    smpl_dataset = smpl_dataset.batch(1)

#    big_dataset = tf.data.Dataset.zip((dataset, smpl_dataset))

#    result = critic([a,b,c], training=True)
#    print(result)

#    init_mean = load_mean_param(config)
    with tf.GradientTape() as gen_tape:
        for image, seg_gt, kps in dataset:
            #image, seg_gt, kps = a
            #features = model(image)
            #print(features.shape)
            #state = tf.concat([features, init_mean], 1)
            #print(state)
            #break
            #pose, shape = b
            #print(image.shape)
            #print(pose.shape)
            #print(shape.shape)
            #print(kps.shape)
            f, axarr = plt.subplots(8,2, figsize=(2, 8), dpi=224)
            plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)
            for j in range(8):
                img = (image[j] + 1) * 0.5
                print(kps.shape)
                ks = ((kps[j, :, :2] + 1) * 0.5) * config.img_size
                ks = ks[:14]
                axarr[j,0].imshow(img)
                axarr[j,0].axis('off')

                seg = tf.concat([seg_gt[j], seg_gt[j], seg_gt[j]], axis=2)
                axarr[j,1].scatter(x=ks[:,0], y=ks[:,1], c=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
                                           'C8', 'C9', 'b', 'g', 'r', 'y'], s=2)
                axarr[j,1].imshow(seg)
                axarr[j,1].axis('off')
            plt.show()



if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    main(config)
