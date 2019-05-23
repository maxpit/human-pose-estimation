from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
from src.config import get_config, prepare_dirs, save_config
from src.data_loader import DataLoader
from src.RunModel import RunModel

def main(config):
    prepare_dirs(config)

    # Load data on CPU
    with tf.device("/cpu:0"):
        data_loader = DataLoader(config)
        image_loader = data_loader.load()
        #smpl_loader = data_loader.get_smpl_loader()

    print(image_loader)

    #trainer = HMRTrainer(config, image_loader, smpl_loader)
    save_config(config)
    #trainer.train()

    images = image_loader['image']
    gts = image_loader['seg_gt']

    print(images)
    print(gts)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.add_subplot(1,2,1)
    plt.imshow(images[0])
    fig.add_subplot(1,2,2)
    plt.imshow(gts[0])
    plt.show()


if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    config.batch_size = 1

    main(config)
