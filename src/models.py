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
#import tensorflow.contrib.slim as slim
import tensorflow.keras.layers as layers
import tensorflow.keras as keras
#from tensorflow.contrib.layers.python.layers.initializers import variance_scaling_initializer
import numpy as np
import math


def Encoder_resnet(is_training=True, weight_decay=0.001, reuse=False):
    """
    Resnet v2-50
    Assumes input is [batch, height_in, width_in, channels]!!
    Input:
    - x: N x H x W x 3
    - weight_decay: float
    - reuse: bool->True if test

    Outputs:
    - cam: N x 3
    - Pose vector: N x 72
    - Shape vector: N x 10
    - variables: tf variables
    """
    import tensorflow.keras.applications as apps
    with tf.name_scope("Encoder_resnet"):
        resnet = apps.ResNet50(include_top=False, weights='imagenet', pooling='avg')
        #resnet.trainable = is_training

    return resnet

    #         net, end_points = resnet_v2.resnet_v2_50(
    #             x,
    #             num_classes=None,
    #             is_training=is_training,
    #             reuse=reuse,
    #             scope='resnet_v2_50')
    #         net = tf.squeeze(net, axis=[1, 2])
    # variables = tf.contrib.framework.get_variables('resnet_v2_50')
    # return net, variables


def Encoder_fc3_dropout(num_input=2133,
                        num_output=85,
                        is_training=True,
                        reuse=False,
                        name="3D_module"):
    """
    3D inference module. 3 MLP layers (last is the output)
    With dropout  on first 2.
    Input:
    - x: N x [|img_feat|, |3D_param|]
    - reuse: bool

    Outputs:
    - 3D params: N x num_output
      if orthogonal: 
           either 85: (3 + 24*3 + 10) or 109 (3 + 24*4 + 10) for factored axis-angle representation
      if perspective:
          86: (f, tx, ty, tz) + 24*3 + 10, or 110 for factored axis-angle.
    - variables: tf variables
    """
    with tf.name_scope(name) as scope:
        model = keras.Sequential()
        model.add(layers.Dense(1024, activation='relu', input_shape=(num_input,)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.5))
        limit = math.sqrt(3.0 * 0.02 / (1024+num_output))
        model.add(layers.Dense(num_output, kernel_initializer=keras.initializers.RandomUniform(-limit, limit)))
        #TODO: check wether weight initializer is same
    return model

    #     net = slim.fully_connected(x, 1024, scope='fc1')
    #     net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout1')
    #     net = slim.fully_connected(net, 1024, scope='fc2')
    #     net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout2')
    #     small_xavier = variance_scaling_initializer(
    #         factor=.01, mode='FAN_AVG', uniform=True)
    #     net = slim.fully_connected(
    #         net,
    #         num_output,
    #         activation_fn=None,
    #         weights_initializer=small_xavier,
    #         scope='fc3')
    #
    # variables = tf.contrib.framework.get_variables(scope)
    # return net, variables

def get_encoder_fn_separate(model_type):
    """
    Retrieves diff encoder fn for image and 3D
    """
    encoder_fn = None
    threed_fn = None
    if 'resnet' in model_type:
        encoder_fn = Encoder_resnet
    else:
        print('Unknown encoder %s!' % model_type)
        exit(1)

    if 'fc3_dropout' in model_type:
        threed_fn = Encoder_fc3_dropout

    if encoder_fn is None or threed_fn is None:
        print('Dont know what encoder to use for %s' % model_type)
        import ipdb
        ipdb.set_trace()

    return encoder_fn, threed_fn

def precompute_C_matrix():
    #   joints:
    #   0: right foot; 1: right knee; 2: right hip; 3: left hip; 4: left knee; 5: left foot; 6: right wrist
    #   7: right elbow; 8: right shoulder; 9: left shoulder; 10: left elbow; 11: left wrist; 12: neck;
    #   13: head; 14-18: head/face
    #
    #   bones:
    #   0: right shin; 1: right thigh; 2: right side; 3: left side; 4: left thigh; 5: left shin; 6: right forearm;
    #   7: right upper arm; 8: right collarbone; 9: left collarbone; 10: left upper arm; 11: left forearm; 12: neck;
    #   13-?: face

    num_bones = 13
    num_joints = 14

    # C = tf.zeros([num_joints, num_bones], tf.int32)
    # C[1, 0] = -1
    # C[0, 0] = 1
    # C[2, 1] = -1
    # C[1, 1] = 1
    # C[8, 2] = -1
    # C[2, 2] = 1
    # C[9, 3] = -1
    # C[3, 3] = 1
    # C[3, 4] = -1
    # C[4, 4] = 1
    # C[4, 5] = -1
    # C[5, 5] = 1
    # C[7, 6] = -1
    # C[6, 6] = 1
    # C[8, 7] = -1
    # C[7, 7] = 1
    # C[12, 8] = -1
    # C[8, 8] = 1
    # C[12, 9] = -1
    # C[9, 9] = 1
    # C[9, 10] = -1
    # C[10, 10] = 1
    # C[10, 11] = -1
    # C[11, 11] = 1
    # C[13, 12] = -1
    # C[12, 12] = 1

    indices_ones = np.arange(num_bones)
    indices_minus_ones = np.array([1, 2, 8, 9, 3, 4, 7, 8, 12, 12, 9, 10, 13])

    C_np = np.zeros([num_joints, num_bones])
    C_np[indices_ones, np.arange(num_bones)] = 1
    C_np[indices_minus_ones, np.arange(num_bones)] = -1
    C = tf.constant(C_np, dtype=tf.float32)

    #with tf.Session() as sess:
    #    c = sess.run(C)
    #    print(c)
    return C

# def Critic_network(
#         joints,
#         shapes,
#         weight_decay,
#         C_matrix,
#         batch_size
# ):
#     """
#     Critic network adapted from
#     "RepNet: Weakly Supervised Training of an Adversarial Reprojection Network for 3D Human Pose Estimation"
#     and Discriminator adapted from
#     "End-to-end Recovery of Human Shape and Pose"
#
#     Input:
#     - joints: N x num_joints x 3
#     - shapes: N x 10
#     - weight_decay: float
#     - C_matrix: num_joints x num_bones
#
#     Outputs:
#     - prediction: N x (1+23) or N x (1+23+1) if do_joint is on.
#     - variables: tf variables
#
#        joints:
#        0: right foot; 1: right knee; 2: right hip; 3: left hip; 4: left knee; 5: left foot; 6: right wrist
#        7: right elbow; 8: right shoulder; 9: left shoulder; 10: left elbow; 11: left wrist; 12: neck;
#        13: head; 14-18: head/face
#
#        bones:
#        0: right shin; 1: right thigh; 2: right side; 3: left side; 4: left thigh; 5: left shin; 6: right forearm;
#        7: right upper arm; 8: right collarbone; 9: left collarbone; 10: left upper arm; 11: left forearm; 12: neck;
#        13-?: face
#     """
#     with tf.name_scope("Critic_network", [joints, shapes]):
#         with tf.variable_scope("Critic") as scope:
#             ##############################################################
#             ### kcs_naive
#             # kcs_naive = tf.Variable(tf.zeros(shape=(1, 13, 13)), trainable=False,
#             #                         dtype=tf.float32)
#             # print('C_matrix', C_matrix)
#             # print('joints', joints)
#             #
#             # joints = joints[:,:14,:]
#             #
#             # for i in range(batch_size):
#             #     kcs_single = tf.matmul(tf.matmul(tf.transpose(C_matrix), tf.matmul(joints[i], tf.transpose(joints[i]))), C_matrix)
#             #     #kcs = tf.assign(kcs[i], tf.cast(kcs_single, tf.float32))
#             #     kcs_naive = tf.concat([kcs_naive, tf.expand_dims(kcs_single, 0)], axis=0)
#             #
#             # kcs_naive = kcs_naive[1:,:,:]
#             # kcs_flattened = tf.reshape(kcs_naive, [-1, 169])
#             #############################################################
#
#             ### kcs_vec
#             joints = joints[:, :14, :]
#             joints_tr = tf.transpose(joints, perm=[0, 2, 1])
#             B = tf.tensordot(joints_tr, C_matrix, 1)
#             B_tr = tf.transpose(B, perm=[0, 2, 1])
#             kcs_long = tf.tensordot(B_tr, tf.transpose(B_tr), 1)
#             kcs_diag = tf.matrix_diag_part(tf.transpose(kcs_long,[1,2,0,3]))
#             kcs = tf.transpose(kcs_diag, [2, 0, 1])
#             kcs_flattened = tf.reshape(kcs, [-1, 169])
#
#             # the first branch operates on the first input
#             first = keras.Sequential()
#             first.add(layers.Dense(100, activation=tf.nn.leaky_relu, input_shape=kcs_flattened.shape[1:]))
#             second = keras.Sequential()
#             joints = tf.reshape(joints, [-1, 42])
#             second.add(layers.Dense(100, activation=tf.nn.leaky_relu, input_shape=joints.shape[1:]))
#
#             merged = tf.concat([first, second], 0)
#             merged.add(layers.Dense(1, input_shape=merged.shape[1:]))
#
#             # the second branch opreates on the second input
#             y = layers.Dense(64, activation="relu")(inputB)
#             y = layers.Dense(32, activation="relu")(y)
#             y = Dense(4, activation="relu")(y)
#             y = Model(inputs=inputB, outputs=y)
#
#             # combine the output of the two branches
#             combined = concatenate([x.output, y.output])
#
#             # apply a FC layer and then a regression prediction on the
#             # combined outputs
#             z = Dense(2, activation="relu")(combined)
#             z = Dense(1, activation="linear")(z)
#
#             # our model will accept the inputs of the two branches and
#             # then output a single value
#             model = Model(inputs=[x.input, y.input], outputs=z)
#
#             critic_model = keras.Sequential()
#             critic_model.add(layers.Dense(100, activation='relu', input_shape=kcs_flattened.shape[1:], ))
#
#
#
#             kcs_fc_out = slim.fully_connected(kcs_flattened, 100,
#                                           activation_fn=tf.nn.leaky_relu,
#                                           scope="kcs_fc")
#
#             joints = tf.reshape(joints, [-1, 42])
#             direct_out = slim.fully_connected(joints, 100,
#                                              activation_fn=tf.nn.leaky_relu,
#                                              scope="direct_fc")
#
#             print('joints', joints)
#             print('kcs', kcs)
#             merged_out = tf.concat([kcs_fc_out, direct_out], 0)
#             print('kcs_fc_out', kcs_fc_out)
#             print('direct_out', direct_out)
#             print('merged_out', merged_out)
#             gan_out = slim.fully_connected(merged_out, 1,
#                                           activation_fn=None,
#                                           scope="wgan")
#             # Do shape on it's own:
#             shapes = slim.stack(
#                shapes,
#                slim.fully_connected, [10, 5],
#                scope="shape_fc1")
#             shape_out = slim.fully_connected(
#                shapes, 1, activation_fn=None, scope="shape_final")
#
#             print('gan_out', gan_out)
#             print('shape_out', shape_out)
#             out = tf.concat([gan_out, shape_out], 1)
#
#             variables = tf.contrib.framework.get_variables(scope)
#             return out, variables

def precompute_C_matrix(num_joints=14):
    # TODO: Check whether LSP order = our order !!!
    """"
    joints:
    0: right foot; 1: right knee; 2: right hip; 3: left hip; 4: left knee; 5: left foot; 6: right wrist
    7: right elbow; 8: right shoulder; 9: left shoulder; 10: left elbow; 11: left wrist; 12: neck;
    13: head; 14-18: head/face

    bones:
    0: right shin; 1: right thigh; 2: right side; 3: left side; 4: left thigh; 5: left shin; 6: right forearm;
    7: right upper arm; 8: right collarbone; 9: left collarbone; 10: left upper arm; 11: left forearm; 12: neck;
    13-?: face
    """

    num_bones = num_joints - 1

    # C = tf.zeros([num_joints, num_bones], tf.int32)
    # C[1, 0] = -1
    # C[0, 0] = 1
    # C[2, 1] = -1
    # C[1, 1] = 1
    # C[8, 2] = -1
    # C[2, 2] = 1
    # C[9, 3] = -1
    # C[3, 3] = 1
    # C[3, 4] = -1
    # C[4, 4] = 1
    # C[4, 5] = -1
    # C[5, 5] = 1
    # C[7, 6] = -1
    # C[6, 6] = 1
    # C[8, 7] = -1
    # C[7, 7] = 1
    # C[12, 8] = -1
    # C[8, 8] = 1
    # C[12, 9] = -1
    # C[9, 9] = 1
    # C[9, 10] = -1
    # C[10, 10] = 1
    # C[10, 11] = -1
    # C[11, 11] = 1
    # C[13, 12] = -1
    # C[12, 12] = 1

    assert num_joints == 14, "num_joints must be 14 for now. Computation of C matrix for more joints will be implemented soon."

    indices_ones = np.arange(num_bones)
    indices_minus_ones = np.array([1, 2, 8, 9, 3, 4, 7, 8, 12, 12, 9, 10, 13])

    C_np = np.zeros([num_joints, num_bones])
    C_np[indices_ones, np.arange(num_bones)] = 1
    C_np[indices_minus_ones, np.arange(num_bones)] = -1
    C = tf.constant(C_np, dtype=tf.float32)

    return C

def get_kcs(joints, C_matrix, num_joints=14):

    joints = joints[:, :num_joints, :]
    joints_tr = tf.transpose(joints, perm=[0, 2, 1])
    B = tf.tensordot(joints_tr, C_matrix, 1)
    B_tr = tf.transpose(B, perm=[0, 2, 1])
    kcs_long = tf.tensordot(B_tr, tf.transpose(B_tr), 1)
    kcs_diag = tf.linalg.diag_part(tf.transpose(kcs_long, [1, 2, 0, 3]))
    kcs = tf.transpose(kcs_diag, [2, 0, 1])
    return kcs

def Critic_network(num_joints=14, use_rotation=False):
    # TODO: Check whether LSP order = our order !!!
    """
    Critic network adapted from
    "RepNet: Weakly Supervised Training of an Adversarial Reprojection Network for 3D Human Pose Estimation"
    and Discriminator adapted from
    "End-to-end Recovery of Human Shape and Pose"

    Input:
    - joints: N x num_joints x 3
    - C_matrix: num_joints x num_bones
    - shapes: N x 10

    Outputs:
    - prediction: N x (1 + 1); one for skeleton, one for shapes

       joints:
       0: right foot; 1: right knee; 2: right hip; 3: left hip; 4: left knee; 5: left foot; 6: right wrist
       7: right elbow; 8: right shoulder; 9: left shoulder; 10: left elbow; 11: left wrist; 12: neck;
       13: head; 14-18: head/face

       bones:
       0: right shin; 1: right thigh; 2: right side; 3: left side; 4: left thigh; 5: left shin; 6: right forearm;
       7: right upper arm; 8: right collarbone; 9: left collarbone; 10: left upper arm; 11: left forearm; 12: neck;
       13-?: face
    """
    print("CRITIC NETWORK")
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

    if use_rotation:
        rotiation_input = layers.input(shape=rotation_input_shape, name="rotation_in")
        rotation_out = layers.Flatten()(rotiation_input)
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

    if use_rotation:
        critic_out = tf.concat([critic_joints_out, shapes_out, rotation_out], 1)
    else:
        critic_out = tf.concat([critic_joints_out, shapes_out], 1)

    if use_rotation:
        model = keras.models.Model(inputs=[kcs_input, joints_input, shapes_input, rotiation_input], outputs=critic_out)
    else:
        model = keras.models.Model(inputs=[kcs_input, joints_input, shapes_input], outputs=critic_out)
    return model

###########################################
# batch_size = 2
#
# head = np.array([0., 0., 3.])
# neck = np.array([0., 0., 2.5])
#
# right_shoulder = np.array([0., -0.5, 2.5])
# right_elbow = np.array([0., -1., 2.5])
# right_wrist = np.array([0.5, -1., 2.5])
# right_hip = np.array([0., -0.5, 1.])
# right_knee = np.array([0.5, -0.5, 0.5])
# right_foot = np.array([0.5, -0.5, 0.])
#
# left_shoulder = np.array([0., 0.5, 2.5])
# left_elbow = np.array([0., 1., 3.])
# left_wrist = np.array([0., 1., 3.5])
# left_hip = np.array([0., 0.5, 1.])
# left_knee = np.array([0.5, 0.5, 0.5])
# left_foot = np.array([-0.5, 0.5, 0.5])
#
# C = precompute_C_matrix()
#
# shapes = None
# weight_decay = None
# joints = np.zeros([batch_size, 14, 3])
# joints[0] = np.stack((right_foot, right_knee, right_hip, left_hip, left_knee, left_foot,
#                    right_wrist, right_elbow, right_shoulder, left_shoulder, left_elbow, left_wrist,
#                    neck, head))
#
# joints[1] = np.stack((right_foot, right_knee + np.array([2, 0, 1]), right_hip, left_hip, left_knee, left_foot,
#                    right_wrist, right_elbow + np.array([4, 2, 9]), right_shoulder, left_shoulder + np.array([2, 2, 2]), left_elbow, left_wrist,
#                    neck, head + np.array([2, 0, 0])))
#
# print("joints.shape: ", joints.shape)
#
# joints_tf = tf.constant(joints, tf.float32)
#
# kcs, kcs_flattened = Critic_network(joints_tf, shapes, weight_decay, C, batch_size)
# print("test")
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     kcs_result = sess.run(kcs)
#     kcs_flattened_result = sess.run(kcs_flattened)
#
#     print("kcs_result.shape: ", kcs_result.shape)
#     print("kcs_result.shape: ", kcs_flattened_result.shape)
#     print("kcs_result = ")
#     print(kcs_result)
#     print("kcs_flattened_result = ")
#     print(kcs_flattened_result)
#     diag = kcs_result[:, np.arange(kcs_result.shape[1]), np.arange(kcs_result.shape[2])]
#     print("bone_lengths: ", np.sqrt(diag))
#     joints_result = sess.run(joints_tf)[0]
#     C_result = sess.run(C)
#     B = np.dot(C_result.T, joints_result)
#     print("joints_result: ", joints_result.shape)
#     print("C_result: ", C_result.shape)
#     print("B: ", B.shape)
#
#     import matplotlib.pyplot as plt
#     from mpl_toolkits.mplot3d import Axes3D
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     plotting = np.concatenate((joints_result[:-1,:], B), axis=1)
#     print(plotting.shape)
#     ax.quiver(plotting[0], plotting[1], plotting[2], plotting[3], plotting[4], plotting[5])
#     ax.set_xlim([-3, 3])
#     ax.set_ylim([-3, 3])
#     ax.set_zlim([-1, 4])
#     plt.show()
###########################################
#
# def Discriminator_separable_rotations(
#         poses,
#         shapes,
#         weight_decay,
# ):
#     """
#     23 Discriminators on each joint + 1 for all joints + 1 for shape.
#     To share the params on rotations, this treats the 23 rotation matrices
#     as a "vertical image":
#     Do 1x1 conv, then send off to 23 independent classifiers.
#
#     Input:
#     - poses: N x 23 x 1 x 9, NHWC ALWAYS!!
#     - shapes: N x 10
#     - weight_decay: float
#
#     Outputs:
#     - prediction: N x (1+23) or N x (1+23+1) if do_joint is on.
#     - variables: tf variables
#     """
#     data_format = "NHWC"
#     with tf.name_scope("Discriminator_sep_rotations", [poses, shapes]):
#         with tf.variable_scope("D") as scope:
#             with slim.arg_scope(
#                 [slim.conv2d, slim.fully_connected],
#                     weights_regularizer=slim.l2_regularizer(weight_decay)):
#                 with slim.arg_scope([slim.conv2d], data_format=data_format):
#                     poses = slim.conv2d(poses, 32, [1, 1], scope='D_conv1')
#                     poses = slim.conv2d(poses, 32, [1, 1], scope='D_conv2')
#                     theta_out = []
#                     for i in range(0, 23):
#                         theta_out.append(
#                             slim.fully_connected(
#                                 poses[:, i, :, :],
#                                 1,
#                                 activation_fn=None,
#                                 scope="pose_out_j%d" % i))
#                     theta_out_all = tf.squeeze(tf.stack(theta_out, axis=1))
#
#                     # Do shape on it's own:
#                     shapes = slim.stack(
#                         shapes,
#                         slim.fully_connected, [10, 5],
#                         scope="shape_fc1")
#                     shape_out = slim.fully_connected(
#                         shapes, 1, activation_fn=None, scope="shape_final")
#                     """ Compute joint correlation prior!"""
#                     nz_feat = 1024
#                     poses_all = slim.flatten(poses, scope='vectorize')
#                     poses_all = slim.fully_connected(
#                         poses_all, nz_feat, scope="D_alljoints_fc1")
#                     poses_all = slim.fully_connected(
#                         poses_all, nz_feat, scope="D_alljoints_fc2")
#                     poses_all_out = slim.fully_connected(
#                         poses_all,
#                         1,
#                         activation_fn=None,
#                         scope="D_alljoints_out")
#                     out = tf.concat([theta_out_all,
#                                      poses_all_out, shape_out], 1)
#
#             variables = tf.contrib.framework.get_variables(scope)
#             return out, variables
