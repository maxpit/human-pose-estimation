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
from glob import glob

from os.path import join, dirname
#import src.util.data_utils as du

def main(config):
    prepare_dirs(config)
    dir_name = '/home/valentin/Code/ADL/human-pose-estimation/data/'
    mpii_dir = join(dir_name, 'upi-s1h/data/mpii/')
    poses_dir = join(mpii_dir, 'poses.npz')
    poses = np.load(poses_dir)['poses']

    all_images = sorted([f for f in glob((mpii_dir+'images/[0-9][0-9][0-9][0-9][0-9].png'))])
    all_seg_gt = sorted([f for f in glob((mpii_dir+'images/[0-9][0-9][0-9][0-9][0-9]_segmentation.png'))])

    for i in range(10):#(poses.shape[2]):
        with tf.io.gfile.GFile(all_images[i], 'rb') as f:
            image_data = f.read()

        with tf.io.gfile.GFile(all_seg_gt[i], 'rb') as f:
            seg_data = f.read()

        seg_gt = tf.image.decode_jpeg(seg_data)
        img = tf.image.decode_jpeg(image_data)
        print("width,height", img.shape)
        print("width,height", seg_gt.shape)
        plt.imshow(img)
        plt.scatter(x=poses[0,:,i], y=poses[1,:,i], c='r', s=4)
        plt.show()



    print(poses.shape)
    print(len(all_seg_gt))
    print(len(all_images))


if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    main(config)
