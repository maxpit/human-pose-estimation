"""
Sets default args

Note all data format is NHWC because slim resnet wants NHWC.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import os.path as osp
from os import makedirs
from glob import glob
from datetime import datetime
import json

import numpy as np

# SMPL loading
curr_path = osp.dirname(osp.abspath(__file__))
model_dir = osp.join(curr_path, '..', 'models')
if not osp.exists(model_dir):
    print('Fix path to models/')
    import ipdb
    ipdb.set_trace()
SMPL_MODEL_PATH = osp.join(model_dir, 'model.pkl')
SMPL_FACE_PATH = osp.join(curr_path, '../src/tf_smpl', 'smpl_faces.npy')
flags.DEFINE_string('smpl_model_path', SMPL_MODEL_PATH,
                    'path to the neurtral smpl model')
flags.DEFINE_string('smpl_face_path', SMPL_FACE_PATH,
                    'path to smpl mesh faces (for easy rendering)')

# General settings --> should not be changed
flags.DEFINE_integer('img_size', 224,
                     'input image size to the network after preprocessing')
flags.DEFINE_string('data_format', 'NHWC', 'Data format')
flags.DEFINE_integer('num_stage', 3, '# of times to iterate through RegressionNetwork')
flags.DEFINE_string('joint_type', 'lsp',
                    'cocoplus (19 keypoints) or lsp 14 keypoints, returned by SMPL')

# Specify the path to your datasets here
DATA_DIR = '/home/valentin/Code/ADL/human-pose-estimation/datasets'
# Setting where to store stuff
flags.DEFINE_string('data_dir', DATA_DIR, 'Where to save training models')
flags.DEFINE_string('logs', 'logs', 'Where to save training models')
flags.DEFINE_string('model_dir', None, 'Where model will be saved -- filled automatically')
# Dataset config
flags.DEFINE_list('datasets', ['lsp_train', 'lsp_ext'],
                          'datasets to use for training')
flags.DEFINE_list('val_datasets', ['lsp_val'],
                          'datasets to use for training')
flags.DEFINE_list('mocap_datasets', ['CMU', 'jointLim'],
                  'datasets to use for adversarial prior training')
# Training settings:
flags.DEFINE_integer('validation_step_size', 50, 'How often to visualize img during training')
flags.DEFINE_integer('log_img_step', 1000, 'How often to visualize img during training')
flags.DEFINE_integer('epoch', 125, '# of epochs to train')
flags.DEFINE_integer('batch_size', 8, 'batch size to use for training loop.')

# Hyper parameters:
# Learning rates
flags.DEFINE_float('generator_lr', 0.0001, 'Encoder learning rate')
flags.DEFINE_float('critic_lr', 0.0005, 'Adversarial prior learning rate')
# Loss weights
flags.DEFINE_float('kpr_loss_weight', 60, 'weight on keypoint reprojection losses')
flags.DEFINE_float('mr_loss_weight', 0.001, 'weight on mesh reprojection loss')
flags.DEFINE_float('critic_loss_weight', 0.01, 'weight on discriminator / critic network')

# Data augmentation settings
flags.DEFINE_integer('trans_max', 20, 'Value to jitter translation')
flags.DEFINE_float('scale_max', 1.23, 'Max value of scale jitter')
flags.DEFINE_float('scale_min', 0.8, 'Min value of scale jitter')

# Model config
flags.DEFINE_boolean('use_mesh_repro_loss', False,
    'specifies whether to use mesh reprojection loss')
flags.DEFINE_boolean('use_kpr_loss', True,
    'specifies whether to use mesh reprojection loss')
flags.DEFINE_boolean('encoder_only', False,
    'if set to True, no adversarial prior is trained --> monsters')
flags.DEFINE_boolean('use_gradient_penalty', True,
    'if set to True, use the gradient penalty for the improved WGAN loss')

# Evaluation config
flags.DEFINE_boolean('do_bone_evaluation', True,
                     'specifies whether to do an evaluation on predicted bone lengths.')
flags.DEFINE_boolean('use_validation', True,
                     'specifies whether to use validation.')

# Checkpoint config
flags.DEFINE_boolean('train_from_checkpoint', False,
                     'if True, train from specified checkpoint')
flags.DEFINE_string('checkpoint_dir', "checkpoints_critic_kp_only_125", 'checkpoint folder')

# Debug mode
flags.DEFINE_boolean('debug', False, 'If set to True, print information that is helpful for debugging.')


def get_config():
    config = flags.FLAGS
    config(sys.argv)

    return config


# ----- For training ----- #


def prepare_dirs(config, prefix=['HMR']):

    postfix = []

    # If config.dataset is not the same as default, add that to name.
    default_dataset = [
        'lsp', 'lsp_ext', 'mpii', 'h36m', 'coco', 'mpi_inf_3dhp'
    ]
    default_mocap = ['CMU', 'H3.6', 'jointLim']

    if sorted(config.datasets) != sorted(default_dataset):
        has_all_default = np.all(
            [name in config.datasets for name in default_dataset])
        if has_all_default:
            new_names = [
                name for name in sorted(config.datasets)
                if name not in default_dataset
            ]
            postfix.append('default+' + '-'.join(sorted(new_names)))
        else:
            postfix.append('-'.join(sorted(config.datasets)))
    if sorted(config.mocap_datasets) != sorted(default_mocap):
        postfix.append('-'.join(config.mocap_datasets))

    if config.num_stage != 3:
        prefix += ["T%d" % config.num_stage]

    postfix.append("Elr%1.e" % config.generator_lr)

    if config.kpr_loss_weight != 1:
        postfix.append("kp-weight%g" % config.kpr_loss_weight)

    if not config.encoder_only:
        postfix.append("Dlr%1.e" % config.critic_lr)
        if config.critic_loss_weight != 1:
            postfix.append("d-weight%g" % config.critic_loss_weight)

    if config.use_mesh_repro_loss:
        postfix.append("mr")

    if config.use_kpr_loss:
        postfix.append("kp")

    prefix.append("_%de_" % config.epoch)

    # Data:
    # Jitter amount:
    if config.trans_max != 20:
        postfix.append("transmax-%d" % config.trans_max)
    if config.scale_max != 1.23:
        postfix.append("scmax_%.3g" % config.scale_max)
    if config.scale_min != 0.8:
        postfix.append("scmin-%.3g" % config.scale_min)

    prefix = '_'.join(prefix)
    postfix = '_'.join(postfix)

    time_str = datetime.now().strftime("%b%d_%H%M")

    save_name = "%s_%s_%s" % (prefix, postfix, time_str)
    config.model_dir = osp.join(config.logs, save_name)

    for path in [config.logs + "_train", config.logs + "_val", config.model_dir]:
        if not osp.exists(path):
            print('making %s' % path)
            makedirs(path)


def save_config(config):
    param_path = osp.join(config.model_dir, "params.json")

    print("[*] MODEL dir: %s" % config.model_dir)
    print("[*] PARAM path: %s" % param_path)

    config_dict = {}
    for k in dir(config):
        config_dict[k] = config.__getattr__(k)

    with open(param_path, 'w') as fp:
        json.dump(config_dict, fp, indent=4, sort_keys=True)
