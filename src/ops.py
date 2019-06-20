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

def find_neighbors_naive(A, B):
    B = tf.cast(B, dtype=tf.int64)
    ind_BA = []
    for i in range(B.shape[0]):
        # print(i)
        distance = tf.reduce_sum(tf.abs(tf.add(A, tf.negative(B[i, :]))), reduction_indices=1)
        # print(distance)
        # Prediction: Get min distance index (Nearest neighbor)
        ind = tf.argmin(distance, 0)
        # with tf.Session() as sess:
        #    print("dist: ", distance.eval())
        #    print("pred: ", ind.eval())
        # print(ind)
        ind_BA.append(ind)
    ind_BA = tf.stack(ind_BA)

    ind_AB = []
    #for i in range(A.shape[0]):
    for i in range(100):
        # print(i)
        distance = tf.reduce_sum(tf.abs(tf.add(B, tf.negative(A[i, :]))), reduction_indices=1)
        # print(distance)
        # Prediction: Get min distance index (Nearest neighbor)
        ind = tf.argmin(distance, 0)
        # with tf.Session() as sess:
        #    print("dist: ", distance.eval())
        #    print("pred: ", ind.eval())
        # print(ind)
        ind_AB.append(ind)
    ind_AB = tf.stack(ind_AB)

    return ind_AB, ind_BA


def bidirectional_dist(A, B):
    # get nearest neighbors B of A
    #if (A = tenso)

    #A_np = np.copy(A)
    #B_np = np.copy(B)
    #print(A_np)
    #print(B_np)
    #B_const = tf.constant(B)
    print("A: ", A)
    print("B: ", B)

    #with tf.Session() as sess:
    #tf.initialize_all_variables().run()
    #print("A.shape: ", A.eval(session=sess).shape)
    #print("B.shape: ", B.eval(session=sess).shape)
    #print("eval A: ", A.eval(session=sess))
    #print("eval B: ", B.eval(session=sess))
    #nbrs_B = NearestNeighbors(n_neighbors=1,
    #                          algorithm='ball_tree').fit(sess.run(A))
    #distances_BA, ind_BA = nbrs_B.kneighbors(sess.run(B))

    # get nearest neighbors A of B
    #nbrs_A = NearestNeighbors(n_neighbors=1,
    #                          algorithm='ball_tree').fit(sess.run(B))
    #distances_AB, ind_AB = nbrs_A.kneighbors(sess.run(A))
    ind_AB, ind_BA = find_neighbors_naive(A, B)

    # compute distances
    dist_BA = tf.norm(tf.to_float(B) - tf.to_float(tf.gather(A, ind_BA)), axis=1)
    dist_AB = tf.norm(tf.to_float(A) - tf.to_float(tf.gather(B, ind_AB)), axis=1)
    summed_dist_BA = tf.reduce_sum(dist_BA)
    summed_dist_AB = tf.reduce_sum(dist_AB)
    # print(distances_BA.sum()+distances_AB.sum())
    # print(tf.math.add(summed_dist_BA, summed_dist_AB).eval())

    return tf.math.add(summed_dist_BA, summed_dist_AB)


def mesh_reprojection_loss(silhouette_gt, silhouette_pred, batch_size, name=None):
    """
    ADL4CV
    Computes bidirectional distance between ground truth silhouette and predicted silhouette
    Inputs:
        silhouette_gt:      ### N x P (num pixels of silhouette) x 2 -->   try with N*P x 3
        silhouette_pred:    N x 6890 (num vertices) x 2


    """
    with tf.name_scope(name, "mesh_reprojection_loss", [silhouette_gt, silhouette_pred]): 
        #K = silhouette_gt.shape[1]
      #  silhouette_gt = tf.reshape(silhouette_gt, (-1, 2))
      #  silhouette_pred = tf.reshape(silhouette_pred, (-1, 2))
      #  sil_gt_np = silhouette_gt.eval()
      #  sil_pred_np = silhouette_pred.eval()
        print("silhouette_gt!!!!!!!!!!!!!!!", silhouette_gt)
        loss = tf.constant(0.) #variable?
        for i in range(batch_size):
            loss = tf.math.add(loss,
                    bidirectional_dist(tf.gather_nd(silhouette_gt,
                                                    tf.where(tf.equal(silhouette_gt[:,0], i)))[:, 1:], silhouette_pred[i,:,:]))
        return loss
