"""
Data loader with data augmentation.
Only used for training.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
from glob import glob

import tensorflow as tf

from .tf_smpl.batch_lbs import batch_rodrigues
from .util import data_utils

_3D_DATASETS = ['h36m', 'up', 'mpi_inf_3dhp']


def num_examples(datasets):
    _NUM_TRAIN = {
        'lsp_few_new': 10,
        'lsp': 1000,
        'lsp_ext': 10000,
        'mpii': 20000,
        'h36m': 312188,
        'coco': 79344,
        'mpi_inf_3dhp': 147221,  # without S8
        # Below is number for MOSH/mocap:
        'H3.6': 1559985,  # without S9 and S11,
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
        self.datasets = config.datasets
        self.batch_size = config.batch_size
        self.data_format = config.data_format
        self.output_size = config.img_size
        self.mocap_datasets = config.mocap_datasets
        # Jitter params:
        self.trans_max = config.trans_max
        self.scale_range = [config.scale_min, config.scale_max]

        self.image_normalizing_fn = data_utils.rescale_image

    def load(self):
        image_loader = self.get_loader()

        return image_loader

    def get_loader(self):
        """
        Outputs:
          image_batch: batched images as per data_format
          label_batch: batched keypoint labels N x K x 3
        """
        files = data_utils.get_all_files(self.dataset_dir, self.datasets)

        do_shuffle = True

        print(files)

        dataset = self.read_data(files)
      
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(self.batch_size)
        
        return dataset 

    def get_smpl_loader(self):
        """
        Loads dataset in form of queue, loads shape/pose of smpl.
        returns a batch of pose & shape
        """

        data_dirs = [
            join(self.dataset_dir, 'mocap_neutrMosh',
                 'neutrSMPL_%s_*.tfrecord' % dataset)
            for dataset in self.mocap_datasets
        ]
        files = []
        for data_dir in data_dirs:
            files += glob(data_dir)

        if len(files) == 0:
            print('Couldnt find any files!!')
            import ipdb
            ipdb.set_trace()

        return self.get_smpl_loader_from_files(files)

    def get_smpl_loader_from_files(self, files):
        """
        files = list of tf records.
        """
        with tf.name_scope('input_smpl_loader'):
            filename_queue = tf.train.string_input_producer(
                files, shuffle=True)

            mosh_batch_size = self.batch_size * self.config.num_stage

            min_after_dequeue = 1000
            capacity = min_after_dequeue + 3 * mosh_batch_size

            pose, shape = data_utils.read_smpl_data(filename_queue)
            pose_batch, shape_batch = tf.train.batch(
                [pose, shape],
                batch_size=mosh_batch_size,
                num_threads=4,
                capacity=capacity,
                name='input_smpl_batch')

            return pose_batch, shape_batch

    def read_data(self, filenames):
        with tf.name_scope(None, 'read_data', filenames): 
            dataset = tf.data.TFRecordDataset(filenames)

            dataset = dataset.map(data_utils.parse_example_proto)
            dataset = dataset.map(self.image_preprocessing)

            return dataset 

    def image_preprocessing(self,
                            image,
                            seg_gt,
                            image_size,
                            label,
                            center,
                            fname,
                            pose=None,
                            gt3d=None):

        print("preprocessing")
        print("img:",image)
        print("seg:",seg_gt)
        print("size:",image_size)
        print("label:",label)
        print("center", center)
        print("pose",pose)
        print("gt3d:",gt3d)
        print(" - ")

        margin = tf.to_int32(self.output_size / 2)
        with tf.name_scope(None, 'image_preprocessing',
                           [image, seg_gt, image_size, label, center]):
            visibility = label[2, :]
            keypoints = label[:2, :]

            # Randomly shift center.
            print('Using translation jitter: %d' % self.trans_max)
            center = data_utils.jitter_center(center, self.trans_max)
            print("shifted center")

            # randomly scale image.
            image, keypoints, center = data_utils.jitter_scale(
                image, image_size, keypoints, center, self.scale_range)
            print("scaled image")

            print("seg prev:", seg_gt)
            seg_gt, _, _ = data_utils.jitter_scale(
                seg_gt, image_size, keypoints, center, self.scale_range)
            print("scaled seg", seg_gt)

            # Pad image with safe margin.
            # Extra 50 for safety.
            margin_safe = margin + self.trans_max + 50
            image_pad = data_utils.pad_image_edge(image, margin_safe)
            seg_gt_pad =  data_utils.pad_image_edge(seg_gt, margin_safe)
            print("padded img and seg", seg_gt_pad)

            center_pad = center + margin_safe
            keypoints_pad = keypoints + tf.to_float(margin_safe)

            start_pt = center_pad - margin

            # Crop image pad.
            start_pt = tf.squeeze(start_pt)
            bbox_begin = tf.stack([start_pt[1], start_pt[0], 0])
            bbox_size = tf.stack([self.output_size, self.output_size, 3])
            bbox_size_gt = tf.stack([self.output_size, self.output_size, 1])

            crop = tf.slice(image_pad, bbox_begin, bbox_size)
            crop_gt = tf.slice(seg_gt_pad, bbox_begin, bbox_size_gt)
            x_crop = keypoints_pad[0, :] - tf.to_float(start_pt[0])
            y_crop = keypoints_pad[1, :] - tf.to_float(start_pt[1])

            crop_kp = tf.stack([x_crop, y_crop, visibility])

            print("before random flip")
            print("crop", crop)
            print("crop_gt", crop_gt)
            print("crop_kp", crop_kp)
            if pose is not None and gt3d is not None:
                crop, crop_gt, crop_kp, new_pose, new_gt3d = data_utils.random_flip(
                    crop, crop_gt, crop_kp, pose, gt3d)
            else:
                crop, crop_gt, crop_kp = data_utils.random_flip(crop, crop_gt, crop_kp)

            # Normalize kp output to [-1, 1]
            final_vis = tf.cast(crop_kp[2, :] > 0, tf.float32)
            final_label = tf.stack([
                2.0 * (crop_kp[0, :] / self.output_size) - 1.0,
                2.0 * (crop_kp[1, :] / self.output_size) - 1.0, final_vis
            ])
            # Preserving non_vis to be 0.
            final_label = final_vis * final_label

            # rescale image from [0, 1] to [-1, 1]
            
            
            ##crop = self.image_normalizing_fn(crop)
            if pose is not None and gt3d is not None:
                return crop, crop_gt, final_label, new_pose, new_gt3d
            else:
                return crop, crop_gt, final_label
