from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags

import tensorflow as tf
import matplotlib.pyplot as plt

import src.config

from src.data_loader import DataLoader

def main(config):
    show_random = False

    # Load data on CPU
    with tf.device("/cpu:0"):
        data_loader = DataLoader(config)
        dataset = data_loader.load()
        smpl_dataset = data_loader.get_smpl_loader()

    if show_random:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(config.batch_size)

    for image, seg_gt, kps in dataset:
        f, axarr = plt.subplots(config.batch_size,2, figsize=(2, config.batch_size), dpi=224)
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)
        for j in range(config.batch_size):
            # undo preprocessing
            img = (image[j] + 1) * 0.5
            ks = ((kps[j, :, :2] + 1) * 0.5) * config.img_size

            # only take first 14 keypoints (no face keyoints) for visualization
            ks = ks[:14]
            # segmentation is stored with only 1 channel but needs 3 channels for vis.
            seg = tf.concat([seg_gt[j], seg_gt[j], seg_gt[j]], axis=2)

            if config.batch_size == 1:
                # draw image
                axarr[0].imshow(img)
                axarr[0].axis('off')
                axarr[1].scatter(x=ks[:,0], y=ks[:,1], c=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
                                           'C8', 'C9', 'b', 'g', 'r', 'y'], s=2)
                axarr[1].imshow(seg)
                axarr[1].axis('off')
            else:
                # draw image
                axarr[j,0].imshow(img)
                axarr[j,0].axis('off')
                axarr[j,1].scatter(x=ks[:,0], y=ks[:,1], c=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
                                           'C8', 'C9', 'b', 'g', 'r', 'y'], s=2)
                axarr[j,1].imshow(seg)
                axarr[j,1].axis('off')

        plt.show()



if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    main(config)
