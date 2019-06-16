"""
TF util operations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
import numpy as np


def keypoint_l1_loss(kp_gt, kp_pred, scale=1., name=None):
    """
    computes: \Sum_i [0.5 * vis[i] * |kp_gt[i] - kp_pred[i]|] / (|vis|)
    Inputs:
      kp_gt  : N x K x 3
      kp_pred: N x K x 2
    """
    with tf.name_scope(name, "keypoint_l1_loss", [kp_gt, kp_pred]):
        kp_gt = tf.reshape(kp_gt, (-1, 3))
        kp_pred = tf.reshape(kp_pred, (-1, 2))

        vis = tf.expand_dims(tf.cast(kp_gt[:, 2], tf.float32), 1)
        res = tf.losses.absolute_difference(kp_gt[:, :2], kp_pred, weights=vis)
        return res


def compute_3d_loss(params_pred, params_gt, has_gt3d):
    """
    Computes the l2 loss between 3D params pred and gt for those data that has_gt3d is True.
    Parameters to compute loss over:
    3Djoints: 14*3 = 42
    rotations:(24*9)= 216
    shape: 10
    total input: 226 (gt SMPL params) or 42 (just joints)

    Inputs:
      params_pred: N x {226, 42}
      params_gt: N x {226, 42}
      # has_gt3d: (N,) bool
      has_gt3d: N x 1 tf.float32 of {0., 1.}
    """
    with tf.name_scope("3d_loss", [params_pred, params_gt, has_gt3d]):
        weights = tf.expand_dims(tf.cast(has_gt3d, tf.float32), 1)
        res = tf.losses.mean_squared_error(
            params_gt, params_pred, weights=weights) * 0.5
        return res


def align_by_pelvis(joints):
    """
    Assumes joints is N x 14 x 3 in LSP order.
    Then hips are: [3, 2]
    Takes mid point of these points, then subtracts it.
    """
    with tf.name_scope("align_by_pelvis", [joints]):
        left_id = 3
        right_id = 2
        pelvis = (joints[:, left_id, :] + joints[:, right_id, :]) / 2.
        return joints - tf.expand_dims(pelvis, axis=1)

def bidirectional_dist(A, B):
    # get nearest neighbors B of A
    #if (A = tenso)
    nbrs_B = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(A.eval())
    distances_BA, ind_BA = nbrs_B.kneighbors(B.eval())

    # get nearest neighbors A of B
    nbrs_A = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(B.eval())
    distances_AB, ind_AB = nbrs_A.kneighbors(A.eval())

    # compute distances
    dist_BA = tf.norm(tf.to_float(B) - tf.to_float(tf.gather(A, ind_BA[:, 0])), axis=1)
    dist_AB = tf.norm(tf.to_float(A) - tf.to_float(tf.gather(B, ind_AB[:, 0])), axis=1)
    summed_dist_BA = tf.reduce_sum(dist_BA)
    summed_dist_AB = tf.reduce_sum(dist_AB)
    # print(distances_BA.sum()+distances_AB.sum())
    # print(tf.math.add(summed_dist_BA, summed_dist_AB).eval())

    return tf.math.add(summed_dist_BA, summed_dist_AB)


def mesh_reprojection_loss(silhouette_gt, silhouette_pred, name=None):
    """
    ADL4CV
    Computes bidirectional distance between ground truth silhouette and predicted silhouette
    Inputs:
        silhouette_gt:      ### N x P (num pixels of silhouette) x 2 -->   try with N*P x 3
        silhouette_pred:    N x 6890 (num vertices) x 2


    """
    with tf.name_scope(name, "mesh_reprojection_loss", [silhouette_gt, silhouette_pred]):
        N = silhouette_gt.shape[0]
        #K = silhouette_gt.shape[1]
      #  silhouette_gt = tf.reshape(silhouette_gt, (-1, 2))
      #  silhouette_pred = tf.reshape(silhouette_pred, (-1, 2))
      #  sil_gt_np = silhouette_gt.eval()
      #  sil_pred_np = silhouette_pred.eval()
        loss = tf.constant(0.)
        for i in range(N):
            loss = tf.math.add(loss, bidirectional_dist(tf.gather_nd(silhouette_gt, tf.where(tf.equal(silhouette_gt[:, 0], 1)))[i, 1:], silhouette_pred[i,:,:]))
        return loss