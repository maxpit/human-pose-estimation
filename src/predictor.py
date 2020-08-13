"""
Predictor
Use a pretrained model to predict SMPL parameters from a 2D image
consisting of [cam (3 - [scale, tx, ty]), pose (72), shape (10)]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .models import CriticNetwork, EncoderNetwork, RegressionNetwork

from .tf_smpl.batch_smpl import SMPL
from .tf_smpl.projection import batch_orth_proj_idrot

import tensorflow as tf
import numpy as np
import os

from os.path import join, dirname
import deepdish as dd

# For drawing
from .util import renderer as vis_util

class Predictor(object):
    def __init__(self, config):
        #######################################################################################
        # Get config information
        #######################################################################################
        self.model_dir = config.model_dir
        self.load_path = config.load_path
        self.data_format = config.data_format
        self.smpl_model_path = config.smpl_model_path
        self.pretrained_model_path = config.pretrained_model_path
        # Data size
        self.img_size = config.img_size
        self.num_stage = config.num_stage
        self.batch_size = config.batch_size
        # Data
        self.checkpoint_dir = config.checkpoint_dir

        self.num_joints = 14

        #######################################################################################
        # Calculate necessary information
        #######################################################################################
        self.proj_fn = batch_orth_proj_idrot

        self.num_cam = 3
        self.num_theta = 72  # 24 * 3
        self.total_params = self.num_theta + self.num_cam + 10

        # Instantiate SMPL
        self.smpl = SMPL(self.smpl_model_path)

        self.renderer = vis_util.SMPLRenderer(
            img_size=self.img_size,
            face_path=config.smpl_face_path)

        self.theta_prev = self.load_mean_param()

        #######################################################################################
        # Set up losses, optimizers and models
        #######################################################################################

        # Initialize optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(0)
        #self.critic_optimizer = tf.keras.optimizers.RMSprop(self.critic_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(0)

        # Load models
        self.image_feature_extractor = EncoderNetwork()
        self.generator3d = RegressionNetwork()
        self.critic_network = CriticNetwork()

        #Restore checkpoint
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.critic_optimizer,
                                         feature_extractor=self.image_feature_extractor,
                                         generator3d=self.generator3d,
                                         discriminator=self.critic_network,
                                         inital_theta=self.theta_prev)
        self.mean_var = self.load_mean_param()
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir)).expect_partial()

    def load_mean_param(self):
        print("LOAD MEAN PARAM THETA")
        mean = np.zeros((1, self.total_params))
        # Initialize scale at 0.9
        mean[0, 0] = 0.9
        mean_path = join(
            dirname(self.smpl_model_path), 'neutral_smpl_mean_params.h5')
        mean_vals = dd.io.load(mean_path)

        mean_pose = mean_vals['pose']
        # Ignore the global rotation.
        mean_pose[:3] = 0.
        mean_shape = mean_vals['shape']

        # This initializes the global pose to be up-right when projected
        mean_pose[0] = np.pi

        mean[0, 3:] = np.hstack((mean_pose, mean_shape))
        mean = tf.constant(mean, tf.float32)
        mean_var = tf.Variable(
            mean, name="mean_param", dtype=tf.float32, trainable=True)
        #init_mean = tf.tile(self.mean_var, [self.batch_size, 1])
        return mean_var


    #@tf.function
    def predict(self, images):
        if self.data_format == 'NCHW':
            # B x H x W x 3 --> B x 3 x H x W
            images = tf.transpose(images, [0, 3, 1, 2])

        all_pred_verts  = []
        all_pred_cams  = []
        all_pred_kps = []
        all_fake_Rs = []

        #Extract feature vector from image using resnet
        extracted_features = self.image_feature_extractor.predict(images)
        theta_prev = tf.tile(self.mean_var, [self.batch_size, 1])
        # Main IEF loop
        for i in range(self.num_stage):
            state = tf.concat([extracted_features, theta_prev], 1)
            delta_theta = self.generator3d.predict(state)

            # Compute new theta
            theta_here = theta_prev + delta_theta
            # cam = N x 3, pose N x self.num_theta, shape: N x 10

            generated_cams = theta_here[:, :self.num_cam]
            generated_poses = theta_here[:, self.num_cam:(self.num_cam + self.num_theta)]
            generated_shapes = theta_here[:, (self.num_cam + self.num_theta):]

            # Rs_wglobal is Nx24x3x3 rotation matrices of poses
            generated_verts, generated_joints, generated_pred_Rs = self.smpl(generated_shapes, generated_poses, get_skin=True)

            all_pred_kps.append(generated_joints)
            all_pred_verts.append(generated_verts)
            all_pred_cams.append(generated_cams)

            # Finally update to end iteration.
            theta_prev = theta_here

        #################################################################################################
        # Return results
        #################################################################################################
        result = {}

        result["generated_joints"] = all_pred_kps[-1]
        result["generated_verts"] = all_pred_verts[-1]
        result["generated_cams"] = all_pred_cams[-1]
        return result

    def predict_single_image(self, image):
        pred_results = self.predict(tf.expand_dims(image, axis=0))

        return pred_results["generated_verts"], pred_results["generated_cams"], pred_results["generated_joints"]
