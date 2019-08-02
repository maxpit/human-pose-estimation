"""
Defines networks.

@Encoder_resnet
@Encoder_resnet_v1_101
@Encoder_fc3_dropout

@Discriminator_separable_rotations

Helper:
@get_encoder_fn_separate
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras as keras
import numpy as np
import math


"""
    EncoderNetwork
    Resnet v2-50 architecture pre-trained on ImageNet without top layers for feature extraction. 
    Assumes input is [batch, height_in, width_in, channels].
    
    Input:
        Image:      N x H x W x 3
    
    Outputs:
        features:   N x num_features (2048)
"""
def EncoderNetwork():

    import tensorflow.keras.applications as apps
    with tf.name_scope("Encoder_resnet"):
        resnet = apps.ResNet50(include_top=False, weights='imagenet', pooling='avg')

    return resnet


"""
    RegressionNetwork
    3D inference module predicting the 85 parameters needed for mesh generation. 
    3 MLP layers (last is the output) with dropout on first 2.
    
    Input:
        x:          N x (num_features (2048) + |theta| (85) )

    Outputs:
        3D params:  N x num_output (85)
                        - 85 parameters Theta: 
                            3 (translation and scale for camera) + 
                            3 (global orientation) + 
                            23*3 (joint orientations) + 
                            10 (shape parameters)
"""
def RegressionNetwork(num_input=2133,
                        num_output=85):

    # build model
    model = keras.Sequential()
    model.add(layers.Dense(1024, activation='relu', input_shape=(num_input,)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))

    # do a small xavier initialization for last layer by hand
    limit = math.sqrt(3.0 * 0.02 / (1024+num_output))
    model.add(layers.Dense(num_output, kernel_initializer=keras.initializers.RandomUniform(-limit, limit)))

    return model


""""
    precompute_C_matrix
    Computes the C matrix yielding information which joints connect to each other. 
    C Matrix is needed for computing the KCS matrix.
    
    Outputs: 
        C_matrix:   num_joints x num_bones
        
    Overview over skeleton configuration:
        joints:
        0: right foot; 1: right knee; 2: right hip; 3: left hip; 4: left knee; 5: left foot; 6: right wrist
        7: right elbow; 8: right shoulder; 9: left shoulder; 10: left elbow; 11: left wrist; 12: neck; 13: head; 
        (14-18: head/face --> not needed)
        bones:
        0: right shin; 1: right thigh; 2: right side; 3: left side; 4: left thigh; 5: left shin; 6: right forearm;
        7: right upper arm; 8: right collarbone; 9: left collarbone; 10: left upper arm; 11: left forearm; 12: neck;
        
    For more information about C matrix / KCS matrix we refer to the paper
    "Repnet: Weakly super-vised training of an adversarial reprojection network for 3dhuman pose estimation."
"""
def precompute_C_matrix(num_joints=14):

    assert num_joints == 14, "num_joints must be 14 for now."
    num_bones = num_joints - 1

    # numpy can be used here as C matrix is only computed once before running the training.
    indices_ones = np.arange(num_bones)
    indices_minus_ones = np.array([1, 2, 8, 9, 3, 4, 7, 8, 12, 12, 9, 10, 13])
    C_np = np.zeros([num_joints, num_bones])
    C_np[indices_ones, np.arange(num_bones)] = 1
    C_np[indices_minus_ones, np.arange(num_bones)] = -1

    # make C a tf.tensor in the end
    C = tf.constant(C_np, dtype=tf.float32)

    return C


"""
    get_kcs
    computes the KCS matrix needed for the critic network.
    
    Input:
        joints:     N x num_joints (14) x 3
        C_matrix:   num_joints x num_bones
"""
def get_kcs(joints, C_matrix, num_joints=14):

    # only take 14 joints (not the face keypoints)
    joints = joints[:, :num_joints, :]

    # compute bone matrix B = X*C (for a single sample from RepNet paper)
    joints_tr = tf.transpose(joints, perm=[0, 2, 1])
    B = tf.tensordot(joints_tr, C_matrix, 1)
    B_tr = tf.transpose(B, perm=[0, 2, 1])

    # compute KCS matrix KCS = B^T * B (for a single sample from RepNet paper)
    # we have more samples given so we do a vectorized computation of all the KCS matrices instead of using for-loops
    kcs_long = tf.tensordot(B_tr, tf.transpose(B_tr), 1)
    kcs_diag = tf.linalg.diag_part(tf.transpose(kcs_long, [1, 2, 0, 3]))
    kcs = tf.transpose(kcs_diag, [2, 0, 1])

    return kcs


"""
    CriticNetwork
    Tries to maximize output scores for real data and to minimize it for fake data.
    
    Input:
        kcs:        N x 13 x 13
        joints:     N x num_joints (14) x 3
        shapes:     N x num_shapes (10)
        rotations:  N x num_joint_rotations (23) x 3 x 3 
                        - 3x3 rotation matrix for each joint rotation
                        - only if use_rotation = True   
                         
    Outputs:
        scores:     N x 3 (2 if use_rotation = False) 
                        - one for skeleton, one for shapes, (one for rotations)
"""
def CriticNetwork(num_joints=14, use_rotation=True):

    # set input shapes
    # for now only 14 joints are possible
    if(num_joints == 14):
        kcs_input_shape = (13, 13)
        joints_input_shape = (14, 3)
        rotation_input_shape = (23, 3, 3)
    elif(num_joints == 19):
        kcs_input_shape = (18, 18)
        joints_input_shape = (19, 3)
        rotation_input_shape = (23, 3, 3)
    else:
        kcs_input_shape, joints_input_shape, rotation_input_shape = None

    # build the network:
    if use_rotation:
        rotation_input = layers.Input(shape=rotation_input_shape, name="rotation_in")
        rotation_out = layers.Flatten()(rotation_input)
        rotation_out = layers.Dense(300, activation=tf.nn.leaky_relu, name="rotation_dense_1")(rotation_out)
        rotation_out = layers.Dense(100, activation=tf.nn.leaky_relu, name="rotation_dense_2")(rotation_out)
        rotation_out = layers.Dense(1, activation=None, name="rotation_dense_3")(rotation_out)

    kcs_input = layers.Input(shape=kcs_input_shape, name="kcs_in")
    kcs_out = layers.Flatten()(kcs_input)
    kcs_out = layers.Dense(100, activation=tf.nn.leaky_relu, name="kcs_dense")(kcs_out)

    joints_input = layers.Input(shape=joints_input_shape)
    joints_out = layers.Flatten()(joints_input)
    joints_out = layers.Dense(100, activation=tf.nn.leaky_relu, name="joints_dense")(joints_out)

    critic_joints_out = tf.concat([kcs_out, joints_out], 1)
    critic_joints_out = layers.Dense(1, input_shape=critic_joints_out.shape[1:], name="combined_dense")(critic_joints_out)

    shapes_input = layers.Input(shape=(10,))
    shapes_out = layers.Dense(10, activation='relu', name="shapes_dense_1")(shapes_input)
    shapes_out = layers.Dense(5, activation='relu', name="shapes_dense_2")(shapes_out)
    shapes_out = layers.Dense(1, name="shapes_dense_3")(shapes_out)

    # concatenate outputs
    if use_rotation:
        critic_out = tf.concat([critic_joints_out, shapes_out, rotation_out], 1)
    else:
        critic_out = tf.concat([critic_joints_out, shapes_out], 1)

    # define model
    if use_rotation:
        model = keras.models.Model(inputs=[kcs_input, joints_input, shapes_input, rotation_input], outputs=critic_out)
    else:
        model = keras.models.Model(inputs=[kcs_input, joints_input, shapes_input], outputs=critic_out)

    return model