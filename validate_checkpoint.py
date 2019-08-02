from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import tensorflow as tf
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

    # Load data on CPU
    with tf.device("/cpu:0"):
        data_loader = DataLoader(config)
        val_dataset = data_loader.load_val_dataset()

    # Make sure that everything is validated both on mr and kp loss
    config.use_mesh_repro_loss = True
    config.use_kp_loss = True
    trainer = Trainer(config, None, None, val_dataset, validation_only=True)
    trainer.validate_checkpoint()

if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    main(config)
