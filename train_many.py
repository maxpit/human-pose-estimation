from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags

import tensorflow as tf

from src.config import get_config, prepare_dirs, save_config
from src.data_loader import DataLoader

from src.trainer import Trainer


def main(config):
    # Uncomment to see device placement
    # tf.debugging.set_log_device_placement(True)

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

    # Prepare directories for saving data
    prepare_dirs(config)

    # Load data on CPU
    with tf.device("/cpu:0"):
        data_loader = DataLoader(config)
        dataset = data_loader.load()
        smpl_loader = data_loader.get_smpl_loader()
        val_dataset = data_loader.load_val_dataset()

    # Train with different settings
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
