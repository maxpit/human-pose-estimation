"""
Data loader with data augmentation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
from glob import glob

import tensorflow as tf

from .tf_smpl.batch_smpl import SMPL
from .util import data_utils

_3D_DATASETS = ['h36m', 'up', 'mpi_inf_3dhp']

def num_examples(datasets):
    _NUM_TRAIN = {
        'lsp_few_new': 10,
        'lsp_few_new_1': 10,
        'lsp_train': 1000,
        'lsp_val': 1000,
        'lsp_ext': 8642,
        'lsp_single': 1,
        'lsp_single_new': 1,
        'single_new_try': 1,
        'lsp_32': 32,
        'CMU': 3934267,
        'jointLim': 181968,
    }

    if not isinstance(datasets, list):
        datasets = [datasets]
    total = 0

    use_dict = _NUM_TRAIN

    for d in datasets:
        total += use_dict[d]
    return total


class DataLoader(object):
    def __init__(self, config):
        self.config = config
        self.dataset_dir = config.data_dir
        self.val_datasets = config.val_datasets
        self.datasets = config.datasets
        self.batch_size = config.batch_size
        self.data_format = config.data_format
        self.output_size = config.img_size
        self.mocap_datasets = config.mocap_datasets
        # Jitter params:
        self.trans_max = config.trans_max
        self.scale_range = [config.scale_min, config.scale_max]
        self.image_normalizing_fn = data_utils.rescale_image
        self.smpl_model_path = config.smpl_model_path
        self.smpl = SMPL(self.smpl_model_path)

    """
    Outputs:
        dataset: Tensorflow dataset containing 2d images, segmentation gt and keypoint gt
    """
    def load(self):
        files = data_utils.get_all_files(self.dataset_dir, self.datasets)
        dataset = self.read_data(files)
        return dataset

    """
    Outputs:
        dataset: Tensorflow dataset containing 2d images, segmentation gt and keypoint gt
    """
    def load_val_dataset(self):
        files = data_utils.get_all_files(self.dataset_dir, self.val_datasets)
        dataset = self.read_data(files)

        return dataset

    """
    Inputs:
        files = list of tf records.
    Outputs:
        dataset = datset containing pose, shape and rotation
    """
    def read_data(self, files):
        dataset = tf.data.TFRecordDataset(files)

        dataset = dataset.map(data_utils.parse_example_proto)
        dataset = dataset.map(self.image_preprocessing)

        return dataset

    """
    Loads dataset in form of queue, loads shape/pose of smpl.
    returns a batch of pose & shape
    """
    def get_smpl_loader(self):
        data_dirs = [
            join(self.dataset_dir, 'mocap_neutrMosh',
                 'neutrSMPL_%s_*.tfrecord' % dataset)
            for dataset in self.mocap_datasets
        ]
        files = []
        for data_dir in data_dirs:
            files += glob(data_dir)

        if len(files) == 0:
            print('At dir', data_dirs)
            print('Couldnt find any files!!')
            import ipdb
            ipdb.set_trace()

        return self.get_smpl_loader_from_files(files)

    """
    Inputs:
        files = list of tf records.
    Outputs:
        mocap_dataset = motion capture datset containing 3d poses
    """
    def get_smpl_loader_from_files(self, files):
        dataset = tf.data.TFRecordDataset(files)
        mocap_dataset = dataset.map(data_utils.parse_mocap_example)
        mocap_dataset = mocap_dataset.map(self.preprocess_poses)
        return mocap_dataset

    """
    This function processes the dataset input data into the correct joint, shape and rotation data
    Inputs:
        pose = SMPL pose parameters
        shpae = SMPL shape paramters
    Outputs:
        joints = joint positions of this SMPL model
        shape = SMPL shape parameters
        rotations = joint rotations of this SMPL model
    """
    def preprocess_poses(self, pose, shape):
        print('preprocess_poses with pose', pose, 'and shape', shape)
        verts, joints, rotations = self.smpl(tf.expand_dims(shape,0), pose, get_skin=True)
        # shape = beta, pose = theta
        return joints, shape, rotations


    """
    This function preprocesses the image and segmentation data
    Inputs:
        image = rgb image in WxHx3
        seg_gt = segmentation data in WxHx1
        image_size = size of the input image
        label = keypoint gt data
        center = center of the human in the image
        fname = filename
    Outputs:
        crop = preprocessed image
        crop_gt = preprocessed segmentation gt
        final_label = preprocessed keypoint data
    """
    def image_preprocessing(self, image, seg_gt, image_size, label, center, fname):

        margin = tf.cast(self.output_size / 2, tf.int32)
        with tf.name_scope('image_preprocessing'):
            visibility = label[2, :]
            keypoints = label[:2, :]

            # Randomly shift center.
            center = data_utils.jitter_center(center, self.trans_max)

            # randomly scale image.
            image, seg_gt, keypoints, center = data_utils.jitter_scale(
                image, seg_gt, image_size, keypoints, center, self.scale_range)

            # Pad image with safe margin.
            # Extra 50 for safety.
            margin_safe = margin + self.trans_max + 50
            image_pad = data_utils.pad_image_edge(image, margin_safe)
            seg_gt_pad =  data_utils.pad_image_edge(seg_gt, margin_safe)

            center_pad = center + margin_safe
            keypoints_pad = keypoints + tf.cast(margin_safe, tf.float32)

            start_pt = center_pad - margin

            # Crop image pad.
            start_pt = tf.squeeze(start_pt)
            bbox_begin = tf.stack([start_pt[1], start_pt[0], 0])
            bbox_size = tf.stack([self.output_size, self.output_size, 3])
            bbox_size_gt = tf.stack([self.output_size, self.output_size, 1])

            crop = tf.slice(image_pad, bbox_begin, bbox_size)
            crop_gt = tf.slice(seg_gt_pad, bbox_begin, bbox_size_gt)
            x_crop = keypoints_pad[0, :] - tf.cast(start_pt[0], tf.float32)
            y_crop = keypoints_pad[1, :] - tf.cast(start_pt[1], tf.float32)

            crop_kp = tf.stack([x_crop, y_crop, visibility])

            crop, crop_gt, crop_kp = data_utils.random_flip(crop, crop_gt, crop_kp)

            # Normalize kp output to [-1, 1]
            final_vis = tf.cast(crop_kp[2, :] > 0, tf.float32)
            final_label = tf.stack([
                2.0 * (crop_kp[0, :] / self.output_size) - 1.0,
                2.0 * (crop_kp[1, :] / self.output_size) - 1.0, final_vis
            ])
            # Preserving non_vis to be 0.
            final_label = final_vis * final_label

            final_label = tf.transpose(final_label)
            # rescale image from [0, 1] to [-1, 1]

            crop = self.image_normalizing_fn(crop)
            return crop, crop_gt, final_label
