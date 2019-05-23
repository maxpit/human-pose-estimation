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
from src.util.load_data import example_run
from src.trainer import HMRTrainer


def main(config):
    prepare_dirs(config)

    # Load data on CPU
    with tf.device("/cpu:0"):
        data_loader = DataLoader(config)
        image_loader = data_loader.load()
        #smpl_loader = data_loader.get_smpl_loader()

    print(image_loader)

    trainer = HMRTrainer(config, image_loader)
    save_config(config)
    trainer.train()

if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    config.batch_size = 10
    main(config)
