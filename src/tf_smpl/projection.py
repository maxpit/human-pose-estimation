""" 
Util functions implementing the reprojection

@batch_orth_proj_idrot
@reproject_vertices
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


"""
    batch_orth_proj_idrot
    Reprojects X to image plane.
    same as applying orth_proj_idrot to each N 
    Inputs:
        X:      N x num_points x 3
        camera: N x 3
"""
def batch_orth_proj_idrot(X, camera, name=None):

    with tf.name_scope("batch_orth_proj_idrot"):

        camera = tf.reshape(camera, [-1, 1, 3], name="cam_adj_shape")

        X_trans = X[:, :, :2] + camera[:, :, 1:]

        shape = tf.shape(X_trans)
        return tf.reshape(
            camera[:, :, 0] * tf.reshape(X_trans, [shape[0], -1]), shape)


"""
    reproject_vertices
    Reprojects verts to image plane.
    Inputs:
        X:          N x num_vertices (6890) x 3
        cam:        N x 3
    Outputs:
        verts_im:   N x 6890 x 2
"""
def reproject_vertices(verts, cam, im_size, name=None):

    with tf.name_scope("mesh_reproject"):
        # reproject to image plane
        verts_reprojected = batch_orth_proj_idrot(verts, cam)

        # get pixel coordinates
        verts_calc = tf.multiply(tf.add(verts_reprojected,
                                          tf.ones_like(verts_reprojected)), 0.5)
        verts_calc = tf.multiply(verts_calc, im_size)

        return verts_calc
