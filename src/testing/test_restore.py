from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
from src.models import Critic_network, Encoder_resnet, Encoder_fc3_dropout, precompute_C_matrix, get_kcs
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
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

print(tf.train.latest_checkpoint(checkpoint_dir))

generator_optimizer = tf.keras.optimizers.Adam(0.00001)
critic_optimizer = tf.keras.optimizers.RMSprop(0.00005)

image_feature_extractor = Encoder_resnet(num_last_layers_to_train=-1)
generator3d = Encoder_fc3_dropout()
critic_network = Critic_network(use_rotation=True)


checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=critic_optimizer,
                                 image_feature_extractor=image_feature_extractor,
                                 generator3d=generator3d,
                                 critic_network=critic_network)

print(checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)))



