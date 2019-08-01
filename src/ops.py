from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


"""
    joint_reprojection_loss
    computes: \Sum_i [0.5 * vis[i] * |kp_gt[i] - kp_pred[i]|] / (|vis|)
    Inputs:
        kp_gt  : N x K x 3
            - N is batch size
            - K is number of joints/keypoints (19)
            - 3 values for x,y position and 1/0 whether the keypoint is visible or not
        kp_pred: N x K x 2
            - N is batch size
            - K is number of joints/keypoints (19)
            - 2 values for x,y
    Outputs:
        summed L1 loss between predicted and ground truth joints for given mesh
"""
def joint_reprojection_loss(kp_gt, kp_pred, scale=1., name="joint_reprojection_loss"):

    with tf.name_scope(name):
        kp_gt = tf.reshape(kp_gt, (-1, 3))
        kp_pred = tf.reshape(kp_pred, (-1, 2))

        # visibilities of joints
        vis = tf.expand_dims(tf.cast(kp_gt[:, 2], tf.float32), 1)

        # only consider distance if keypoint is visible
        res = tf.compat.v1.losses.absolute_difference(kp_gt[:, :2], kp_pred, weights=vis)

        return res


"""
    find_nearest_neighbors
    Computes nearest neighbors between two sets of 2-dim vectors.
    To avoid for loops a distance matrix is computed.
    Inputs:
        A:    num_A x 2
        B:    num_B x 2
    Outputs:
        indices of nearest neighbors of A to B and vice versa
"""
def find_nearest_neighbors(A, B):

    # compute L2 distance matrix between A and B
    dists = tf.scalar_mul(-2., tf.matmul(A, B, transpose_b=True)) + \
            tf.expand_dims(tf.reduce_sum(tf.square(A), 1), 1) + \
            tf.expand_dims(tf.reduce_sum(tf.square(B), axis=1), 0)

    # get the indices of matrices A and B where distance is smallest (nearest neighbor indices)
    ind_AB = tf.argmin(dists, 1)
    ind_BA = tf.argmin(dists, 0)

    return ind_AB, ind_BA


"""
    bidirectional_dist
    Computes bidirectional distance between two sets of 2-dim vectors.
    Inputs:
        A:    num_A x 2
        B:    num_B x 2
    Outputs:
        bidirectional distance
"""
def bidirectional_dist(A, B):

    # get nearest neighbors between A and B
    ind_AB, ind_BA = find_nearest_neighbors(A, B)

    # compute distances
    # distance of matrix B to its nearest neighbors in A
    # we take L2 loss here
    dist_BA = tf.norm(B - tf.gather(A, ind_BA), axis=1)
    #dist_AB = tf.norm(tf.to_float(A) - tf.to_float(tf.gather(B, ind_AB)), axis=1)

    # distance of matrix A to its nearest neighbors in B
    # we take L1 loss here
    dist_AB = tf.reduce_sum(tf.abs(A - tf.gather(B, ind_AB)), axis=1)

    # sum up all dists
    summed_dist_BA = tf.reduce_sum(dist_BA)
    summed_dist_AB = tf.reduce_sum(dist_AB)

    return tf.math.add(summed_dist_BA, summed_dist_AB)


"""
    mesh_reprojection_loss
    Computes mesh reprojection loss for a given batch.
    Inputs:
        silhouette_gt:      N*Pi x 3 
                                - first column contains index of image in batch
                                - second and third column are x- and y-image coordinates
                                - Pi is number of silhouette pixels in image i
        silhouette_pred:    N x 6890 (num vertices mesh) x 2
    Outputs:
        Mesh reprojection loss on batch
"""
def mesh_reprojection_loss(silhouette_gt, silhouette_pred, batch_size, name="mesh_reprojection_loss"):

    with tf.name_scope(name):
        for i in range(batch_size):

            # get ground truth silhouette of ith image in batch in shape [Pi x 2]
            silhouette_gt_x = tf.gather_nd(silhouette_gt, tf.where(tf.equal(silhouette_gt[:, 0], i)))[:, 2]
            silhouette_gt_y = tf.gather_nd(silhouette_gt, tf.where(tf.equal(silhouette_gt[:, 0], i)))[:, 1]
            silhouette_points_gt = tf.stack([silhouette_gt_x, silhouette_gt_y], axis=1)

            # compute the bidirectional distance of image i
            bi_loss = bidirectional_dist(silhouette_points_gt, silhouette_pred[i, :, :])
            bi_loss_scaled = bi_loss/(silhouette_gt.shape[1] +
                                      silhouette_pred.shape[1])

            # sum up losses for all images
            if i == 0:
                loss = bi_loss_scaled
            else:
                loss = loss + bi_loss_scaled
        return loss