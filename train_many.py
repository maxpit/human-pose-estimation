from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np

#import skimage.io as io
import tensorflow as tf
import matplotlib.pyplot as plt

#from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config as config
from src.config import get_config, prepare_dirs, save_config
from src.data_loader import DataLoader
#from src.RunModel import RunModel
#from src.util.load_data import example_run
from src.trainer import Trainer

#import src.util.data_utils as du

def main(config):
#    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
    prepare_dirs(config)

    # Load data on CPU
    with tf.device("/cpu:0"):
        data_loader = DataLoader(config)
        dataset = data_loader.load()
        smpl_loader = data_loader.get_smpl_loader()
        val_dataset = data_loader.load_val_dataset()

#    iterator = dataset.make_one_shot_iterator()

    config.encoder_only = False
    config.use_mesh_repro_loss = True
    config.use_kp_loss = False

    prepare_dirs(config)
    trainer = Trainer(config, dataset, smpl_loader, val_dataset)
    save_config(config)
    trainer.train()

    config.encoder_only = False
    config.use_mesh_repro_loss = False
    config.use_kp_loss = True

    prepare_dirs(config)
    trainer = Trainer(config, dataset, smpl_loader, val_dataset)
    save_config(config)
    trainer.train()

    config.encoder_only = False
    config.use_mesh_repro_loss = True
    config.use_kp_loss = True

    prepare_dirs(config)
    trainer = Trainer(config, dataset, smpl_loader, val_dataset)
    save_config(config)
    trainer.train()

if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    main(config)
