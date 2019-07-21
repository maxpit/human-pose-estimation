"""
HMR trainer.
From an image input, trained a model that outputs 85D latent vector
consisting of [cam (3 - [scale, tx, ty]), pose (72), shape (10)]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .data_loader import num_examples

from .ops import keypoint_l1_loss, compute_3d_loss, align_by_pelvis, mesh_reprojection_loss
from .models import Critic_network, Encoder_resnet, Encoder_fc3_dropout, precompute_C_matrix, get_kcs

from .tf_smpl.batch_lbs import batch_rodrigues
from .tf_smpl.batch_smpl import SMPL
from .tf_smpl.projection import batch_orth_proj_idrot
from .tf_smpl.projection import reproject_vertices

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops.losses import losses

import time
from datetime import datetime
import tensorflow as tf
import numpy as np
import os

from tensorflow.python.ops import resources
from tensorflow.python.ops import variables

from os.path import join, dirname
import deepdish as dd

# For drawing
from .util import renderer as vis_util
#from .util.data_utils import get_silhouette_from_seg_im as get_sil

class HMRTrainer(object):
    def __init__(self, config, dataset, mocap_dataset = None, val_dataset=None):
        """
        Args:
          config
          if no 3D label is available,
              data_loader is a dict
          else
              data_loader is a dict
        mocap_dataset is a tuple (pose, shape)
        """

        #######################################################################################
        # Get config information
        #######################################################################################
        self.model_dir = config.model_dir
        self.load_path = config.load_path
        self.data_format = config.data_format
        self.smpl_model_path = config.smpl_model_path
        self.pretrained_model_path = config.pretrained_model_path
        self.encoder_only = config.encoder_only
        self.use_3d_label = config.use_3d_label
        self.use_rotation = True #config.use_rotation
        self.use_validation = False#config.use_validation
        # Data size
        self.img_size = config.img_size
        #print(self.img_size)
        self.num_stage = config.num_stage
        self.batch_size = config.batch_size
        self.max_epoch = config.epoch
        self.num_gen_steps_per_itr = config.num_gen_steps_per_itr
        self.use_mesh_repro_loss = config.use_mesh_repro_loss
        self.use_kp_loss= config.use_kp_loss
        # Data
        num_images = num_examples(config.datasets)
        num_mocap = num_examples(config.mocap_datasets)
        self.val_step = config.validation_step_size
        self.log_img_step = config.log_img_step
        self.validation_step_size = config.validation_step_size
        # Model spec
        self.model_type = config.model_type
        # Weight decay
        self.e_wd = config.e_wd
        self.d_wd = config.d_wd

        # Gather loss weights
        self.generator_kp_loss_weight = config.generator_loss_weight
        self.critic_loss_weight = config.critic_loss_weight
        self.e_3d_weight = config.e_3d_weight
        self.mr_loss_weight = config.mr_loss_weight

        # Optimizer, learning rate
        self.generator_lr = config.generator_lr
        self.critic_lr = config.critic_lr

        self.use_gradient_penalty = config.use_gradient_penalty
        self.num_joints = 14
        self.do_bone_evaluation = False

        #######################################################################################
        # Calculate necessary information
        #######################################################################################

        # Calculate C Matrix for the KCS Layer 
        self.C = precompute_C_matrix()

        self.proj_fn = batch_orth_proj_idrot

        self.num_cam = 3
        self.num_theta = 72  # 24 * 3
        self.total_params = self.num_theta + self.num_cam + 10

        self.num_itr_per_epoch = num_images / self.batch_size
        if self.num_itr_per_epoch < 1:
            self.num_itr_per_epoch = 1

        self.num_mocap_itr_per_epoch = num_mocap / self.batch_size

        # For visualization:
        num2show = np.minimum(6, self.batch_size)
        # Take half from front & back
        self.show_these = tf.constant(
            np.hstack(
                [np.arange(num2show / 2), self.batch_size - np.arange(3) - 1]),
            tf.int32)

        # Instantiate SMPL
        self.smpl = SMPL(self.smpl_model_path)

        self.renderer = vis_util.SMPLRenderer(
            img_size=self.img_size,
            face_path=config.smpl_face_path)

        # Initialise the tensorboard writers
        self.training_writer = tf.summary.create_file_writer(self.model_dir + 'training')
        self.val_writer = tf.summary.create_file_writer(self.model_dir+'validation')

        self.theta_prev = self.load_mean_param()
#        if self.use_3d_label:
#            self.poseshape_loader = data_loader['label3d']
#            # image_loader[3] is N x 2, first column is 3D_joints gt existence,
#            # second column is 3D_smpl gt existence
#            self.has_gt3d_joints = data_loader['has3d'][:, 0]
#            self.has_gt3d_smpl = data_loader['has3d'][:, 1]

        #######################################################################################
        # Print train information
        #######################################################################################

        print('model dir: %s', self.model_dir)
        print('load path: %s', self.load_path)
        print('data_format: %s', self.data_format)
        print('smpl_model_path: %s', self.smpl_model_path)
        print('pretrained_model_path: %s', self.pretrained_model_path)
        print('encoder only:', self.encoder_only)
        print('use_3d_label:', self.use_3d_label)
        print('image_size:', self.img_size)
        print('num_stage:', self.num_stage)
        print('batch_size:', self.batch_size)
        print('num_images: ', num_images)
        print('num_mocap', num_mocap)

        #######################################################################################
        # Load data sets 
        #######################################################################################

        # Create train and validation dataset from dataset with given train/val
        # split

        self.full_dataset = []
#TODO no validation at the moment...
#        if config.use_validation:
#            num_train_samples = int(num_images * config.train_val_split)
#
#            print("NUM_TRAIN_SAMPLES:",num_train_samples)
#            train_set = dataset.take(num_train_samples).shuffle(buffer_size=10000).repeat()
#            val_set = dataset.skip(num_train_samples).shuffle(buffer_size=10000).repeat()
#
#            train_set = train_set.batch(self.batch_size)
#            val_set = val_set.batch(self.batch_size)
#
#            self.full_dataset.append(train_set)
#        else:
        dataset = dataset.shuffle(buffer_size=10000).repeat()
        dataset = dataset.batch(self.batch_size)
        self.full_dataset.append(dataset)

        if not self.encoder_only:
            critic_dataset = mocap_dataset.shuffle(buffer_size=1000).repeat()
            critic_dataset = critic_dataset.batch(self.batch_size*3)
            self.full_dataset.append(critic_dataset)
        else:
            self.full_dataset.append(tf.data.Dataset.range(2000).repeat())

        if val_dataset is not None:
            val_dataset = val_dataset.shuffle(buffer_size=1000).repeat()
            val_dataset = val_dataset.batch(self.batch_size)
            self.full_dataset.append(val_dataset)
        else:
            self.full_dataset.append(tf.data.Dataset.range(2000).repeat())
        # create one dataset so its easier to iterate over it
        self.full_dataset = tf.data.Dataset.zip(tuple(self.full_dataset))

        #######################################################################################
        # Set up losses, optimizers and models
        #######################################################################################

        # Initialize optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(self.generator_lr)
        self.critic_optimizer = tf.keras.optimizers.RMSprop(self.critic_lr)
        #self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)

        # Load models
        self.image_feature_extractor = Encoder_resnet()
        self.generator3d = Encoder_fc3_dropout()
        self.critic_network = Critic_network(use_rotation=self.use_rotation)

        #self.checkpoint_dir = './training_checkpoints'
        #self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        #self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
        #                                 discriminator_optimizer=self.critic_optimizer,
        #                                 feature_extractor=self.image_feature_extractor,
        #                                 generator3d=self.generator3d,
        #                                 discriminator=self.critic_network)

    def use_pretrained(self):
        """
        Returns true only if:
          1. model_type is "resnet"
          2. pretrained_model_path is not None
          3. model_dir is NOT empty, meaning we're picking up from previous
             so fuck this pretrained model.
        """
        if ('resnet' in self.model_type) and (self.pretrained_model_path is
                                              not None):
            # Check is model_dir is empty
            import os
            if os.listdir(self.model_dir) == []:
                return True

        return False


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


    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def val_step(self, images, seg_gts, kp2d_gts):
        print("not autographed")

        if self.data_format == 'NCHW':
            # B x H x W x 3 --> B x 3 x H x W
            images = tf.transpose(images, [0, 3, 1, 2])
            seg_gts = tf.transpose(seg_gts, [0, 3, 1, 2])

        kp_losses = []
        mr_losses = []
        all_pred_verts  = []
        all_pred_cams  = []
        all_pred_kps = []
        generator_critic_losses = []
        all_fake_Rs = []

        #Extract feature vector from image using resnet
        extracted_features = self.image_feature_extractor(images, training=False)
        theta_prev = tf.tile(self.mean_var, [self.batch_size, 1])
        # Main IEF loop
        for i in range(self.num_stage):
            state = tf.concat([extracted_features, theta_prev], 1)
            delta_theta = self.generator3d(state, training=False)

            # Compute new theta
            theta_here = theta_prev + delta_theta
            # cam = N x 3, pose N x self.num_theta, shape: N x 10

            generated_cams = theta_here[:, :self.num_cam]
            generated_poses = theta_here[:, self.num_cam:(self.num_cam + self.num_theta)]
            generated_shapes = theta_here[:, (self.num_cam + self.num_theta):]

            # Rs_wglobal is Nx24x3x3 rotation matrices of poses
            generated_verts, generated_joints, generated_pred_Rs = self.smpl(generated_shapes, generated_poses, get_skin=True)
            generated_pred_Rs = generated_pred_Rs[:,1:,:]

            #For visualization
            all_pred_verts.append(tf.gather(generated_verts, self.show_these))
            all_pred_cams.append(tf.gather(generated_cams, self.show_these))

            ##############################################################################################
            # Calculate Generator Losses
            ##############################################################################################

            #Calculate keypoint reprojection loss
            pred_kp = batch_orth_proj_idrot(generated_joints, generated_cams,
                                                name='val_proj2d_stage%d' % i)
            # For visulalization
            all_pred_kps.append(tf.gather(pred_kp, self.show_these))
            if self.use_kp_loss:
                kp_losses.append(
                    self.generator_kp_loss_weight * keypoint_l1_loss(kp2d_gts, pred_kp)
                )

            #Calculate mesh reprojection loss
            if self.use_mesh_repro_loss:
                silhouette_pred = reproject_vertices(generated_verts,
                                                     generated_cams,
                                                     tf.constant([self.img_size, self.img_size], tf.float32),
                                                     name='val_mesh_reproject_stage%d' % i)
                # silhouette_gt: first entry = index sample;
                #                second,third = coordinate of pixel with value > 0.
                silhouette_gt = tf.cast(tf.where(tf.greater(seg_gts, 0.))[:, :3], tf.float32)

                repro_loss = mesh_reprojection_loss(
                    silhouette_gt, silhouette_pred, self.batch_size,
                    name='mesh_repro_loss%d' % i)
                repro_loss_scaled = repro_loss * self.mr_loss_weight

                mr_losses.append(repro_loss_scaled)

            #Calculate ctritic loss
            if not self.encoder_only:
                all_fake_Rs.append(generated_pred_Rs)
                kcs = get_kcs(generated_joints, self.C)
                #generated_joints = tf.transpose(generated_joints, perm=[0,2,1])[:,:,:self.num_joints]
                generator_critic_out = self.critic_network([kcs,
                                                   generated_joints[:,:self.num_joints,:],
                                                   generated_shapes,
                                                   generated_pred_Rs],
                                                   training=False
                                                 )

                generator_critic_loss = - tf.reduce_sum(tf.reduce_mean(generator_critic_out, 0))
                #tf.print("CRITIC LOSS", generator_critic_loss)
                generator_critic_losses.append(generator_critic_loss * self.critic_loss_weight)

            # Save things for visualiations:
            #self.all_verts.append(tf.gather(verts, self.show_these))
            #if(not self.use_mesh_repro_loss):
            #self.all_pred_kps.append(tf.gather(pred_kp, self.show_these))
            #if(self.use_mesh_repro_loss):
            #    self.all_pred_silhouettes.append(tf.gather(silhouette_pred, self.show_these))
            #self.all_pred_cams.append(tf.gather(cams, self.show_these))

            # Finally update to end iteration.
            theta_prev = theta_here

        generator_loss_sum = 0.
        if self.use_kp_loss:
            #generator_loss_sum.append(kp_losses[-1])
            generator_loss_sum += kp_losses[-1]
        if self.use_mesh_repro_loss:
            generator_loss_sum += mr_losses[-1]
        if not self.encoder_only:
            generator_loss_sum += generator_critic_losses[-1]
            pass


        #################################################################################################
        # Return results
        #################################################################################################
        result = {}


        if self.use_kp_loss:
            result["kp_losses"] = kp_losses
        if self.use_mesh_repro_loss:
            result["mr_losses"] = mr_losses
        all_pred_kps = tf.stack(all_pred_kps, axis=1)
        result["generated_kps"] = all_pred_kps
        all_pred_cams = tf.stack(all_pred_cams, axis=1)
        all_pred_verts = tf.stack(all_pred_verts, axis=1)
        result["generated_verts"] = all_pred_verts
        result["generated_cams"] = all_pred_cams

        if not self.encoder_only:
            result["generator_critic_losses"] = generator_critic_losses

        return result


    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(self, images, seg_gts, kp2d_gts, joint3d_gt, shape_gts, Rs_gt, step):
        print("not autographed")

        #############################################################################################
        # Generator update
        #############################################################################################


        #tf.print("batched_images", len(batched_images))
        #tf.print("batched_seg_gts", len(batched_seg_gts))
        #tf.print("batched_kp2d_gts", len(batched_kp2d_gts))
        #tf.print("images", images)
        #tf.print("seg_gts", seg_gts)
        #tf.print("kp2d_gts", kp2d_gts)
        #tf.print("joint3d_gt", joint3d_gt)
        #tf.print("shape_gts", shape_gts)
        #tf.print("step", self.global_step)

        #tf.print("images", images)
        # First make sure data_format is right
        if self.data_format == 'NCHW':
            # B x H x W x 3 --> B x 3 x H x W
            images = tf.transpose(images, [0, 3, 1, 2])
            seg_gts = tf.transpose(seg_gts, [0, 3, 1, 2])

        fake_joints = []
        fake_shapes = []
        kp_losses = []
        mr_losses = []
        all_pred_verts  = []
        all_pred_cams  = []
        all_pred_kps = []
        all_pred_silhouettes = []
        generator_critic_losses = []
        all_fake_Rs = []

        with tf.GradientTape() as gen_tape:
            #Extract feature vector from image using resnet
            #print("images.shape", images.shape)
            extracted_features = self.image_feature_extractor(images, training=True)
            theta_prev = tf.tile(self.mean_var, [self.batch_size, 1])
            #theta_prev = tf.zeros((self.batch_size, 85))
            #print("extracted_features.shape", extracted_features.shape)
            #print("theta_prev.shape", theta_prev.shape)
            # Main IEF loop
            for i in range(self.num_stage):
            #for i in np.arange(1):
                #tf.print('iteration', i)
                #print("theta_prev", self.theta_prev)
                state = tf.concat([extracted_features, theta_prev], 1)
                #print(state.shape)
                #TODO how do i get reuse=true for this?
                if i != self.num_stage-1:
                    delta_theta = self.generator3d(state, training=False)
                else:
                    delta_theta = self.generator3d(state, training=True)#, reuse=True)

                # Compute new theta
                theta_here = theta_prev + delta_theta
                # cam = N x 3, pose N x self.num_theta, shape: N x 10

                generated_cams = theta_here[:, :self.num_cam]
                generated_poses = theta_here[:, self.num_cam:(self.num_cam + self.num_theta)]
                generated_shapes = theta_here[:, (self.num_cam + self.num_theta):]
                fake_shapes.append(generated_shapes)

                # Rs_wglobal is Nx24x3x3 rotation matrices of poses
                generated_verts, generated_joints, generated_pred_Rs = self.smpl(generated_shapes, generated_poses, get_skin=True)
                generated_pred_Rs = generated_pred_Rs[:,1:,:]
                fake_joints.append(generated_joints)

                #For visualization
                all_pred_verts.append(tf.gather(generated_verts, self.show_these))
                all_pred_cams.append(tf.gather(generated_cams, self.show_these))

                ##############################################################################################
                # Calculate Generator Losses
                ##############################################################################################

                #Calculate keypoint reprojection loss
                pred_kp = batch_orth_proj_idrot(generated_joints, generated_cams,
                                                    name='proj2d_stage%d' % i)
                # For visulalization
                all_pred_kps.append(tf.gather(pred_kp, self.show_these))
                if self.use_kp_loss:
                    kp_losses.append(
                        self.generator_kp_loss_weight * keypoint_l1_loss(kp2d_gts, pred_kp)
                    )

                #Calculate mesh reprojection loss
                if self.use_mesh_repro_loss:
                    silhouette_pred = reproject_vertices(generated_verts,
                                                         generated_cams,
                                                         tf.constant([self.img_size, self.img_size], tf.float32),
                                                         name='mesh_reproject_stage%d' % i)
                    # silhouette_gt: first entry = index sample; 
                    #                second,third = coordinate of pixel with value > 0.
                    silhouette_gt = tf.cast(tf.where(tf.greater(seg_gts, 0.))[:, :3], tf.float32)

                    repro_loss = mesh_reprojection_loss(
                        silhouette_gt, silhouette_pred, self.batch_size,
                        name='mesh_repro_loss%d' % i)
                    repro_loss_scaled = repro_loss * self.mr_loss_weight

                    mr_losses.append(repro_loss_scaled)

                #Calculate 3d joint loss
                #TODO not used atm
                if self.use_3d_label:
                    loss_poseshape, loss_joints = self.get_3d_loss(
                        pred_Rs, shapes, joints)
                    loss_3d_params.append(loss_poseshape)
                    loss_3d_joints.append(loss_joints)

                #Calculate ctritic loss
                if not self.encoder_only:
                    all_fake_Rs.append(generated_pred_Rs)
                    kcs = get_kcs(generated_joints, self.C)
                    #generated_joints = tf.transpose(generated_joints, perm=[0,2,1])[:,:,:self.num_joints]
                    generator_critic_out = self.critic_network([kcs,
                                                       generated_joints[:,:self.num_joints,:],
                                                       generated_shapes,
                                                       generated_pred_Rs],
                                                       training=False
                                                     )

                    generator_critic_loss = - tf.reduce_sum(tf.reduce_mean(generator_critic_out, 0))
                    #tf.print("CRITIC LOSS", generator_critic_loss)
                    generator_critic_losses.append(generator_critic_loss * self.critic_loss_weight)

                # Save things for visualiations:
                #self.all_verts.append(tf.gather(verts, self.show_these))
                #if(not self.use_mesh_repro_loss):
                #self.all_pred_kps.append(tf.gather(pred_kp, self.show_these))
                #if(self.use_mesh_repro_loss):
                #    self.all_pred_silhouettes.append(tf.gather(silhouette_pred, self.show_these))
                #self.all_pred_cams.append(tf.gather(cams, self.show_these))

                # Finally update to end iteration.
                theta_prev = theta_here

                #theta_here = theta_prev + delta_theta
                #print("theta here", theta_here)
                #gts = tf.ones_like(theta_here)
                #print("gts", gts)
                #loss = tf.linalg.norm(theta_here-gts) # works

                #gt_joints = tf.ones_like(generated_joints)
                #gt_verts = tf.ones_like(generated_verts)
                # loss = tf.linalg.norm(generated_joints-gt_joints) + tf.linalg.norm(generated_verts-gt_verts) # works

                #kp_gt = tf.ones_like(pred_kp)
                #loss = tf.linalg.norm(pred_kp-kp_gt) # works

                #loss = self.generator_kp_loss_weight * keypoint_l1_loss(small_kps, pred_kp) # works

            #loss = kp_losses[-1]
            #print(loss)

            #grads = gen_tape.gradient(loss,
            #                  self.generator3d.trainable_variables)
            #print("grads", grads)
            #self.generator_optimizer.apply_gradients(zip(grads,
            #                                             self.generator3d.trainable_variables))
            #print("applied gradients")

            ###########################################################################################
            # Generator optimization
            ###########################################################################################

            #if not self.encoder_only:
            #    #print("critic loss", critic_loss)
            #    generator_critic_loss = (-self.critic_loss_weight * critic_loss)

            ##TODO no 3d labels used yet
            #if self.use_3d_label:
            #    self.generator_loss_3d = loss_3d_params[-1]
            #    self.generator_loss_3d_joints = loss_3d_joints[-1]

            #  self.generator_loss += (self.generator_loss_3d + self.generator_loss_3d_joints)

            #all_train_vars = self.generator3d.trainable_variables + self.image_feature_extractor.trainable_variables

            #print("generator_loss", generator_loss)
            #print("gen_tape", gen_tape)
            #print('trainable_variables encoder', self.image_feature_extractor.trainable_variables)
            #print('trainable_variables generator3d', self.generator3d.trainable_variables)

            variables = self.image_feature_extractor.trainable_variables + self.generator3d.trainable_variables
            variables.append(self.mean_var)

            #tf.print("variables generator len", len(self.generator3d.trainable_variables))
            #tf.print("variables extractor len", len(self.image_feature_extractor.trainable_variables))
            #tf.print("variables mean va", self.mean_var)
            #tf.print("variables len", len(variables))

            generator_loss_sum = 0.
            if self.use_kp_loss:
                #generator_loss_sum.append(kp_losses[-1])
                generator_loss_sum += kp_losses[-1]
            if self.use_mesh_repro_loss:
                generator_loss_sum += mr_losses[-1]
            if not self.encoder_only:
                generator_loss_sum += generator_critic_losses[-1]
                pass

            #tf.print("gen loss sum", generator_loss_sum)
            gradients_of_generator = gen_tape.gradient(generator_loss_sum, variables)
            #tf.print("gradients generator length", len(gradients_of_generator))
            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, variables))


            #tf.print("APPLIED GRADIENTS GENERATOR =)")
            #print("step %g: time %g, generator_loss: %.4f" %(step, 0, generator_loss))

            #self.global_step = self.global_step + 1

        #############################################################################################
        # Critic update
        #############################################################################################

        #critic_network_loss = []
        if not self.encoder_only:
            with tf.GradientTape() as critic_tape:
                all_fake_joints = tf.concat(fake_joints, 0)[:,:self.num_joints,:]
                all_fake_Rs = tf.concat(all_fake_Rs, 0)
                all_fake_shapes = tf.concat(fake_shapes, 0)
                joint3d_gt = joint3d_gt[:,:self.num_joints,:]
                #tf.print('real_joints', joint3d_gt)
                #tf.print('real_shapes', shape_gts)
                #tf.print('fake_joints', fake_joints)
                #tf.print('fake_shapes', fake_shapes)
                #tf.print('all_fake_joints', all_fake_joints)
                #tf.print('all_fake_shapes', all_fake_shapes)

                real_kcs = get_kcs(joint3d_gt, self.C)
                real_output = self.critic_network([real_kcs, joint3d_gt, shape_gts, Rs_gt], training=True)

                fake_kcs = get_kcs(all_fake_joints, self.C)
                fake_output = self.critic_network([fake_kcs,
                                                   all_fake_joints,
                                                   all_fake_shapes,
                                                   all_fake_Rs],
                                                   training=True)

                ##########################################
                ### Try WGAN loss                      ###
                ##########################################
                #fake_output = -fake_output
                #critic_network_loss = tf.reduce_sum(real_output * fake_output)/real_output.shape[0]

                #critic_network_loss.append(tf.reduce_sum(tf.reduce_mean(fake_output - real_output, 0)))
                critic_network_loss = tf.reduce_sum(tf.reduce_mean(fake_output - real_output, 0))

                if self.use_gradient_penalty:
                    alpha = tf.random.uniform(all_fake_joints.shape)
                    beta = tf.random.uniform(all_fake_shapes.shape)
                    gamma = tf.random.uniform(all_fake_Rs.shape)
                    interpolated_joints = all_fake_joints + alpha * (joint3d_gt - all_fake_joints)
                    interpolated_kcs = get_kcs(all_fake_joints, self.C)
                    interpolated_shapes = all_fake_shapes + beta * (shape_gts - all_fake_shapes)
                    interpolated_Rs = all_fake_Rs + gamma * (Rs_gt - all_fake_Rs)
                    out_interpolated = self.critic_network([interpolated_kcs,
                                                            interpolated_joints[:, :self.num_joints, :],
                                                            interpolated_shapes,
                                                            interpolated_Rs],
                                                           training=True)
                    gradients_to_penalize = tf.gradients(ys=out_interpolated,
                                                         xs=[interpolated_kcs,
                                                             interpolated_joints,
                                                             interpolated_shapes,
                                                             interpolated_Rs])
                    #tf.print("out_interpolated", out_interpolated.shape)
                    #tf.print("gradients", len(gradients_to_penalize))
                    #tf.print("gradients[0]", gradients_to_penalize[0].shape)
                    #tf.print("gradients[1]", gradients_to_penalize[1].shape)
                    #tf.print("gradients[2]", gradients_to_penalize[2].shape)
                    #tf.print("gradients[0].norm",
                    #         tf.norm(tf.reduce_mean(gradients_to_penalize[0]), ord='euclidean'))
                    #tf.print("gradients[1].norm",
                    #         tf.norm(tf.reduce_mean(gradients_to_penalize[1]), ord='euclidean'))
                    #tf.print("gradients[2].norm",
                    #         tf.norm(tf.reduce_mean(gradients_to_penalize[2]), ord='euclidean'))
                    penalty_1 = tf.square(
                        1. - tf.norm(tf.reduce_mean(gradients_to_penalize[0], 0), ord='euclidean'))
                    penalty_2 = tf.square(
                        1. - tf.norm(tf.reduce_mean(gradients_to_penalize[1], 0), ord='euclidean'))
                    penalty_3 = tf.square(
                        1. - tf.norm(tf.reduce_mean(gradients_to_penalize[2], 0), ord='euclidean'))
                    penalty_4 = tf.square(
                        1. - tf.norm(tf.reduce_mean(gradients_to_penalize[3], 0), ord='euclidean'))
                    penalty = (penalty_1 + penalty_2 + penalty_3 + penalty_4)
                    #tf.print("penalty", penalty)

                    #critic_network_loss.append(penalty)
                    critic_network_loss = critic_network_loss + 10. * penalty
                ##########################################
                ##########################################
                #tf.print("critic_network_loss:", critic_network_loss)

                #critic_network_loss = -(tf.reduce_sum(real_output - fake_output) / real_output.shape[0])

            gradients_of_discriminator = critic_tape.gradient(critic_network_loss,
                                                            self.critic_network.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                        self.critic_network.trainable_variables))
            #tf.print("variables critic network length:", len(self.critic_network.trainable_variables))
            #tf.print("gradients critic length:", len(gradients_of_discriminator))
            #tf.print("critic network loss:", critic_network_loss)
            #tf.print("APPLIED GRADIENTS CRITIC :-)")
            #print("step %g: time %g, generator_loss: %.4f" %(step, 0, critic_network_loss))


        #################################################################################################
        # Return results
        #################################################################################################
        result = {}


        if self.use_kp_loss:
            result["kp_losses"] = kp_losses
        if self.use_mesh_repro_loss:
            result["mr_losses"] = mr_losses
        all_pred_kps = tf.stack(all_pred_kps, axis=1)
        result["generated_kps"] = all_pred_kps
        all_pred_cams = tf.stack(all_pred_cams, axis=1)
        all_pred_verts = tf.stack(all_pred_verts, axis=1)
        result["generated_verts"] = all_pred_verts
        result["generated_cams"] = all_pred_cams

        if not self.encoder_only:
            result["generator_critic_losses"] = generator_critic_losses
            result["critic_penalty"] = penalty
            result["critic_network_loss"] = critic_network_loss

        if self.do_bone_evaluation:
            bones_pred = tf.linalg.diag_part(get_kcs(all_fake_joints, self.C))
            avg_total_bone_length_pred = tf.reduce_mean(tf.reduce_sum(bones_pred, axis=1))
            result["avg_total_bone_length_pred"] = avg_total_bone_length_pred

            bones_gt = tf.linalg.diag_part(get_kcs(joint3d_gt, self.C))
            avg_total_bone_length_gt = tf.reduce_mean(tf.reduce_sum(bones_gt, axis=1))
            result["avg_total_bone_length_gt"] = avg_total_bone_length_gt


        return result

    def visualize_img(self, img, gt_kp, vert, pred_kp, cam, renderer, seg_gt=None):
        """
        Overlays gt_kp and pred_kp on img.
        Draws vert with text.
        Renderer is an instance of SMPLRenderer.
        """
        gt_vis = gt_kp[:, 2].astype(bool)
        loss = np.sum((gt_kp[gt_vis, :2] - pred_kp[gt_vis])**2)
        debug_text = {"sc": cam[0], "tx": cam[1], "ty": cam[2], "kpl": loss}
        # Fix a flength so i can render this with persp correct scale
        f = 5.
        tz = f / cam[0]
        cam_for_render = 0.5 * self.img_size * np.array([f, 1, 1])
        cam_t = np.array([cam[1], cam[2], tz])
        # Undo pre-processing.
        input_img = (img + 1) * 0.5
        rend_img = renderer(vert + cam_t, cam_for_render, img=input_img)
        rend_img = vis_util.draw_text(rend_img, debug_text)

        # Draw skeleton
        gt_joint = ((gt_kp[:, :2] + 1) * 0.5) * self.img_size
        pred_joint = ((pred_kp + 1) * 0.5) * self.img_size
        img_with_gt = vis_util.draw_skeleton(
            input_img, gt_joint, draw_edges=False, vis=gt_vis)
        skel_img = vis_util.draw_skeleton(img_with_gt, pred_joint)

        if(self.use_mesh_repro_loss):
            # seg gt needs to be same dimension as color image.
            seg_gt = seg_gt.squeeze()
            seg2_gt = np.stack((seg_gt,seg_gt,seg_gt), axis=2)
            rend_seg_gt = renderer(vert + cam_t, cam_for_render, img=seg2_gt)
            combined = np.hstack([skel_img, rend_img / 255., rend_seg_gt / 255.])
        else:
            combined = np.hstack([skel_img, rend_img / 255.])
        return combined

    def draw_results(self, imgs, segs_gt, gt_kps, est_verts, pred_kps, cam, step):
        import io
        import matplotlib.pyplot as plt
        import cv2

        imgs = imgs.numpy()
        segs_gt = segs_gt.numpy()
        gt_kps = gt_kps.numpy()
        est_verts = est_verts.numpy()
        pred_kps = pred_kps.numpy()
        cam = cam.numpy()

        if self.data_format == 'NCHW':
            imgs = np.transpose(imgs, [0, 2, 3, 1])

        img_summaries = []

        if not self.use_mesh_repro_loss:
            segs_gt = np.empty_like(imgs)

        for img_id, (img, seg_gt, gt_kp, verts, keypoints, cams) in enumerate(
                zip(imgs, segs_gt, gt_kps, est_verts, pred_kps, cam)):
            # verts, joints, cams are a list of len T.
            all_rend_imgs = []
            for vert, joint, cam in zip(verts, keypoints, cams):

                if(self.use_mesh_repro_loss):
                    rend_img = self.visualize_img(img, gt_kp, vert, joint, cam,
                                                  self.renderer, seg_gt)
                else:
                    rend_img = self.visualize_img(img, gt_kp, vert, joint, cam,
                                                  self.renderer)
                all_rend_imgs.append(rend_img)
            combined = np.vstack(all_rend_imgs)

            buf = io.BytesIO()

            plt.imsave(buf, combined, format='png')
            buf.seek(0)

            # Convert PNG buffer to TF image
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            # Add the batch dimension
            image = tf.expand_dims(image, 0)

            tf.summary.image(("vis_images/%d" % img_id), image, step=step)

            buf.flush()
            buf.close()
            plt.close()

    def train(self):
        print('started training')
        # For rendering!

        print('...')
        self.mean_var = self.load_mean_param()
        self.global_step = tf.Variable(1, name='global_step', trainable=False, dtype=tf.int64)
        step = tf.Variable(1, name='step', trainable=False, dtype=tf.int64)

        itr = 0
        epoch = 0
        for data_gen, data_critic, val_data in self.full_dataset:
            images, segmentations, keypoints = data_gen
            if not self.encoder_only:
                joints, shapes, rotations = data_critic
                joints = tf.squeeze(joints, axis=1)
            else:
                joints = None
                rotations = None
                shapes = None

            start_time = time.time()
            result = self.train_step(images, segmentations, keypoints, joints, shapes,
                                     tf.squeeze(rotations)[:,1:,:], step)
            end_time = time.time()

            step = step + 1
            ############################################################################################
            # Write Generator update to tensorboard
            ############################################################################################

            with self.training_writer.as_default():

                if self.use_kp_loss:
                    tf.summary.scalar('generator/kp_loss', result["kp_losses"][-1], step=step)
                if self.use_mesh_repro_loss:
                    tf.summary.scalar('generator/mr_loss', result["mr_losses"][-1], step=step)

                if tf.equal((step % self.log_img_step), tf.constant(0, dtype=tf.int64)):
                    show_images = tf.gather(images, self.show_these)
                    show_segs = tf.gather(segmentations, self.show_these)
                    show_kps = tf.gather(keypoints, self.show_these)
                    print("drawing images now")
                    # self.draw_results(show_images, show_segs, show_kps,
                    #                   result["generated_verts"], result["generated_kps"],
                    #                   result["generated_cams"], step)

                ##############################################################################################
                # Write Critic update to tensorboard
                ############################################################################################
                if not self.encoder_only:
                    tf.summary.scalar('critic/critic_network_loss', result["critic_network_loss"], step=step)
                    tf.summary.scalar('critic/generator_critic_loss',
                                      result["generator_critic_losses"][-1], step=step)
                    tf.summary.scalar('critic/penalty', result["critic_penalty"], step=step)
                if self.do_bone_evaluation:
                    tf.summary.scalar('avg_total_bone_lenth_pred', result["avg_total_bone_length_pred"], step=step)
                    tf.summary.scalar('avg_total_bone_lenth_gt', result["avg_total_bone_length_gt"], step=step)
                self.training_writer.flush()

            print("one step took", (end_time - start_time), "seconds")

            if self.use_validation:
                ##########################################################################
                # validation step
                ##########################################################################
                images, segmentations, keypoints = val_data

                val_result = self.val_step(images, segmentations, keypoints)

                with self.val_writer.as_default():
                    if self.use_kp_loss:
                        tf.summary.scalar('generator/kp_loss', val_result["kp_losses"][-1], step=step)
                    if self.use_mesh_repro_loss:
                        tf.summary.scalar('generator/mr_loss', val_result["mr_losses"][-1], step=step)

                    if tf.equal((step % self.log_img_step), tf.constant(0, dtype=tf.int64)):
                        show_images = tf.gather(images, self.show_these)
                        show_segs = tf.gather(segmentations, self.show_these)
                        show_kps = tf.gather(keypoints, self.show_these)
                        print("drawing images now")
                        # self.draw_results(show_images, show_segs, show_kps,
                        #                   val_result["generated_verts"], val_result["generated_kps"],
                        #                   val_result["generated_cams"], step)
                    self.val_writer.flush()

            itr += self.num_gen_steps_per_itr
            print("itr", itr, "/", self.num_itr_per_epoch)
            if itr >= self.num_itr_per_epoch:
                itr = 0
                epoch += 1

                #self.checkpoint.save(self.checkpoint_prefix)

                if epoch >= self.max_epoch:
                    break

                print("epoch", epoch)
        print('Finish training on %s' % self.model_dir)
