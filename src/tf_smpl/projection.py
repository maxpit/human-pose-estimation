""" 
Util functions implementing the camera

@@batch_orth_proj_idrot
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

def batch_orth_proj_idrot(X, camera, name=None):
    """
    X is N x num_points x 3
    camera is N x 3
    same as applying orth_proj_idrot to each N 
    """
    with tf.name_scope(name, "batch_orth_proj_idrot", [X, camera]):
        # TODO check X dim size.
        # tf.Assert(X.shape[2] == 3, [X])
        print("camera: ", camera)
        print("X: ", X)

        camera = tf.reshape(camera, [-1, 1, 3], name="cam_adj_shape")

        X_trans = X[:, :, :2] + camera[:, :, 1:]

        shape = tf.shape(X_trans)
        return tf.reshape(
            camera[:, :, 0] * tf.reshape(X_trans, [shape[0], -1]), shape)


def reproject_vertices(proc_param, verts, cam, im_size):
    """
    TODO: adapt to multiple samples
    """
    # im_size = img.shape[:2]
    print("cam repro: ", cam)
    cam_ = tf.expand_dims(cam, 0)
    verts_ = tf.expand_dims(verts, 0)
    verts_projected = batch_orth_proj_idrot(tf.cast(verts_, tf.float32), tf.cast(cam_, tf.float32))
    verts_projected = ((verts_projected + 1) * 0.5 * proc_param['scale']) * im_size  # proc_param['img_size'] instead of im_size?

    img_size = proc_param['img_size']
    margin = int(img_size / 2)
    undo_scale = 1. / np.array(proc_param['scale'])
    verts_original = (verts_projected + proc_param['start_pt'] - margin) * undo_scale
    #verts_pixel = tf.cast(verts_original, tf.int32)
    return verts_original  # shape: num_verts x 2