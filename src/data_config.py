"""
Sets default args
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import os.path as osp

curr_path = osp.dirname(osp.abspath(__file__))
SMPL_FACE_PATH = osp.join(curr_path, '../src/tf_smpl', 'smpl_faces.npy')

file_dir = osp.join(curr_path, 'data')

lsp_dir = osp.join(file_dir, 'lsp')
lsp_e_dir = osp.join(file_dir, 'lspet_dataset/')
mpii_dir = osp.join(file_dir, 'upi-s1h/data/mpii/')

flags.DEFINE_string('lsp_dir', lsp_dir, 'path to lsp dataset')
flags.DEFINE_string('lsp_e_dir', lsp_e_dir, 'path to lsp extended dataset')
flags.DEFINE_string('lsp_im', osp.join(lsp_dir, 'images/'), 'path to lsp images')
flags.DEFINE_string('lsp_e_im', osp.join(lsp_e_dir, 'images/'), 'path to lsp_ext images')
flags.DEFINE_string('lsp_seg', osp.join(file_dir, 'upi-s1h/data/lsp/'), 'path to lsp segmentations')
flags.DEFINE_string('lsp_e_seg', osp.join(file_dir, 'upi-s1h/data/lsp_extended/'), 'path to lsp_ext segmentations')
flags.DEFINE_string('mpii_dir', mpii_dir, 'path to mpiii dataset')
flags.DEFINE_string('mpii_poses_dir', osp.join(mpii_dir, 'poses.npz'), 'path to mpii dir')

flags.DEFINE_bool('create_lsp', False, 'True if dataset should be created')
flags.DEFINE_bool('create_lsp_val', False, 'True if dataset should be created')
flags.DEFINE_bool('create_lsp_ext', False, 'True if dataset should be created')
flags.DEFINE_bool('create_mpii', False, 'True if dataset should be created')

dataset_dir = osp.join(curr_path, 'datasets')
