"""
Trainer.
From an image input, trained a model that outputs 85D latent vector
consisting of [cam (3 - [scale, tx, ty]), orientation (global orientation 3, joint orientations 69), shape (10)]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .data_loader import num_examples

from .ops import joint_reprojection_loss, mesh_reprojection_loss, compute_gradient_penalty
from .models import CriticNetwork, EncoderNetwork, RegressionNetwork, precompute_C_matrix, get_kcs

from .tf_smpl.batch_smpl import SMPL
from .tf_smpl.projection import batch_orth_proj_idrot
from .tf_smpl.projection import reproject_vertices

import time
import tensorflow as tf
import numpy as np
import os

from os.path import join, dirname
import deepdish as dd

# For drawing
from .util import renderer as vis_util


class Trainer(object):

    """
    Input:
        config
        dataset:            tuple (image, segmentation gt, keypoint gt)
        mocap_dataset:      tuple (joints, shape, rotations)
        val_dataset:        tuple (image, segmentation gt, keypoint gt)
        validation_only:    if set to True, other datasets are not loaded
                            and tensorboard files for training are not created.
    """
    def __init__(self, config, dataset = None,
                               mocap_dataset = None,
                               val_dataset=None,
                               validation_only=False):
        #######################################################################################
        # Get config information
        #######################################################################################
        self.model_dir = config.model_dir
        self.logs = config.logs
        self.load_path = config.load_path
        self.data_format = config.data_format
        self.smpl_model_path = config.smpl_model_path
        self.encoder_only = config.encoder_only
        self.use_validation = config.use_validation
        self.train_from_checkpoint = config.train_from_checkpoint

        # Data size
        self.img_size = config.img_size
        self.num_stage = config.num_stage
        self.batch_size = config.batch_size
        self.max_epoch = config.epoch
        self.use_mesh_repro_loss = config.use_mesh_repro_loss
        self.use_kpr_loss = config.use_kpr_loss

        # Data
        num_images = num_examples(config.datasets)
        num_mocap = num_examples(config.mocap_datasets)
        self.val_step_size = config.validation_step_size
        self.log_img_step = config.log_img_step
        self.checkpoint_dir = config.checkpoint_dir

        # Gather loss weights
        self.kpr_loss_weight = config.kpr_loss_weight
        self.critic_loss_weight = config.critic_loss_weight
        self.mr_loss_weight = config.mr_loss_weight

        # Optimizer, learning rate
        self.generator_lr = config.generator_lr
        self.critic_lr = config.critic_lr

        # Use the gradient penalty for the improved WGAN loss.
        # Should be set to True as no gradient clipping is provided.
        self.use_gradient_penalty = config.use_gradient_penalty

        # Do additional bone length evaluation
        self.do_bone_evaluation = config.do_bone_evaluation

        # Number of jonints used for critic network --> has to be 14 for now
        self.num_joints = 14

        # Enable debugging
        self.debug = config.debug

        #######################################################################################
        # Calculate necessary information
        #######################################################################################

        # Calculate C Matrix for the KCS Layer 
        self.C = precompute_C_matrix()

        self.proj_fn = batch_orth_proj_idrot

        self.num_cam = 3
        self.num_theta = 72  # 1*3 (global) + 23 * 3
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

        self.theta_prev = self.load_mean_param()

        if not validation_only:
            # Initialise the tensorboard writers
            self.training_writer = tf.summary.create_file_writer(self.model_dir + 'training')
            self.val_writer = tf.summary.create_file_writer(self.model_dir+'validation')

        #######################################################################################
        # Print train information
        #######################################################################################

        print('model dir: %s', self.model_dir)
        print('load path: %s', self.load_path)
        print('data_format: %s', self.data_format)
        print('smpl_model_path: %s', self.smpl_model_path)
        print('encoder only:', self.encoder_only)
        print('image_size:', self.img_size)
        print('num_stage:', self.num_stage)
        print('batch_size:', self.batch_size)
        print('num_images: ', num_images)
        print('num_mocap', num_mocap)

        #######################################################################################
        # Load data sets 
        #######################################################################################
        self.full_dataset = []

        if dataset is not None:
            dataset = dataset.shuffle(buffer_size=10000).repeat()
            dataset = dataset.batch(self.batch_size)
            self.full_dataset.append(dataset)

        if mocap_dataset is not None:
            critic_dataset = mocap_dataset.shuffle(buffer_size=10000).repeat()
            critic_dataset = critic_dataset.batch(self.batch_size*self.num_stage)
            self.full_dataset.append(critic_dataset)

        if val_dataset is not None:
            if not validation_only:
                val_dataset = val_dataset.shuffle(buffer_size=1000).repeat()
            self.val_dataset = val_dataset.batch(self.batch_size)
            self.full_dataset.append(self.val_dataset)
        else:
            # add dummy data
            self.full_dataset.append(tf.data.Dataset.range(2000).repeat())

        # create one dataset so its easier to iterate over it
        self.full_dataset = tf.data.Dataset.zip(tuple(self.full_dataset))

        #######################################################################################
        # Set up losses, optimizers and models
        #######################################################################################

        # Initialize optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(self.generator_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)

        # Load models
        self.image_feature_extractor = EncoderNetwork()
        self.generator3d = RegressionNetwork()
        self.critic_network = CriticNetwork()

        print("checkpoint")
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.critic_optimizer,
                                         feature_extractor=self.image_feature_extractor,
                                         generator3d=self.generator3d,
                                         discriminator=self.critic_network,
                                         inital_theta=self.theta_prev)


    def load_mean_param(self):
        if self.debug:
            print("load mean parameter")
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
        return mean_var


    def val_step(self, images, seg_gts, kp2d_gts):
        print("not autographed")

        if self.data_format == 'NCHW':
            # B x H x W x 3 --> B x 3 x H x W
            images = tf.transpose(images, [0, 3, 1, 2])
            seg_gts = tf.transpose(seg_gts, [0, 3, 1, 2])

        kpr_losses = []
        mr_losses = []
        all_pred_verts  = []
        all_pred_cams  = []
        all_pred_kps = []
        generator_critic_losses = []
        all_fake_Rs = []

        # extract feature vector from image using resnet
        extracted_features = self.image_feature_extractor.predict(images)

        # get initial theta
        theta_prev = tf.tile(self.mean_var, [self.batch_size, 1])

        # Main regression loop
        for i in range(self.num_stage):
            state = tf.concat([extracted_features, theta_prev], 1)
            delta_theta = self.generator3d.predict(state)

            # Compute new theta
            theta_here = theta_prev + delta_theta

            # get cams, joint orientations and shapes
            generated_cams = theta_here[:, :self.num_cam]
            generated_joint_orientations = theta_here[:, self.num_cam:(self.num_cam + self.num_theta)]
            generated_shapes = theta_here[:, (self.num_cam + self.num_theta):]

            # SMPL generator yields the 3D coordinates of the 6890 vertices, the 19 keypoints/joints
            # and the rotations of the 24 rotation matrices
            generated_verts, generated_joints, generated_pred_Rs = self.smpl(generated_shapes, generated_joint_orientations, get_skin=True)
            generated_pred_Rs = generated_pred_Rs[:,1:,:]

            # for visualization
            all_pred_verts.append(tf.gather(generated_verts, self.show_these))
            all_pred_cams.append(tf.gather(generated_cams, self.show_these))

            ##############################################################################################
            # Calculate Generator Losses
            ##############################################################################################

            # calculate keypoint reprojection loss
            pred_kp = batch_orth_proj_idrot(generated_joints, generated_cams,
                                                name='val_proj2d_stage%d' % i)
            # For visulalization
            all_pred_kps.append(tf.gather(pred_kp, self.show_these))
            kpr_losses.append(
                self.kpr_loss_weight * joint_reprojection_loss(kp2d_gts, pred_kp,
                                                                 name='val_kp_loss')
            )

            # calculate mesh reprojection loss
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
                    name='val_mesh_repro_loss%d' % i)
                repro_loss_scaled = repro_loss * self.mr_loss_weight

                mr_losses.append(repro_loss_scaled)

            # calculate ctritic loss
            if not self.encoder_only:
                all_fake_Rs.append(generated_pred_Rs)
                kcs = get_kcs(generated_joints, self.C)

                generator_critic_out = self.critic_network([kcs,
                                                   generated_joints[:,:self.num_joints,:],
                                                   generated_shapes,
                                                   generated_pred_Rs],
                                                   training=False
                                                 )

                generator_critic_loss = - tf.reduce_sum(tf.reduce_mean(generator_critic_out, 0))
                generator_critic_losses.append(generator_critic_loss * self.critic_loss_weight)

                if self.debug:
                    tf.print("generator critic loss (validation)", generator_critic_loss)

            # finally update to end iteration.
            theta_prev = theta_here

        # compute overall generator validation loss
        generator_loss_sum = 0.
        if self.use_kpr_loss:
            generator_loss_sum += kpr_losses[-1]
        if self.use_mesh_repro_loss:
            generator_loss_sum += mr_losses[-1]
        if not self.encoder_only:
            generator_loss_sum += generator_critic_losses[-1]

        #################################################################################################
        # Return results
        #################################################################################################
        result = {}

        result["kpr_losses"] = kpr_losses
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


    # the use of `tf.function` causes the function to be "compiled" in tf2.
    @tf.function
    def train_step(self, images, seg_gts, kp2d_gts, joint3d_gt, shape_gts, Rs_gt, step):

        print("not autographed")
        if self.debug:
            tf.print("images", images)
            tf.print("seg_gts", seg_gts)
            tf.print("kp2d_gts", kp2d_gts)
            tf.print("joint3d_gt", joint3d_gt)
            tf.print("shape_gts", shape_gts)
            tf.print("step", self.global_step)

        #############################################################################################
        # Generator update
        #############################################################################################

        # First make sure data_format is right
        if self.data_format == 'NCHW':
            # B x H x W x 3 --> B x 3 x H x W
            images = tf.transpose(images, [0, 3, 1, 2])
            seg_gts = tf.transpose(seg_gts, [0, 3, 1, 2])

        fake_joints = []
        fake_shapes = []
        kpr_losses = []
        mr_losses = []
        all_pred_verts  = []
        all_pred_cams  = []
        all_pred_kps = []
        generator_critic_losses = []
        all_fake_Rs = []

        with tf.GradientTape() as gen_tape:

            # extract feature vector from image using resnet
            extracted_features = self.image_feature_extractor(images, training=True)

            # get initial theta
            theta_prev = tf.tile(self.mean_var, [self.batch_size, 1])

            # Main regression loop
            for i in range(self.num_stage):
                state = tf.concat([extracted_features, theta_prev], 1)

                if i != self.num_stage-1:
                    delta_theta = self.generator3d(state, training=False)
                else:
                    delta_theta = self.generator3d(state, training=True)

                # Compute new theta
                theta_here = theta_prev + delta_theta

                # get cams, joint orientations and shapes
                generated_cams = theta_here[:, :self.num_cam]
                generated_joint_orientations = theta_here[:, self.num_cam:(self.num_cam + self.num_theta)]
                generated_shapes = theta_here[:, (self.num_cam + self.num_theta):]
                fake_shapes.append(generated_shapes)

                # SMPL generator yields the 3D coordinates of the 6890 vertices, the 19 keypoints/joints
                # and the rotations of the 24 rotation matrices
                generated_verts, generated_joints, generated_pred_Rs = self.smpl(generated_shapes, generated_joint_orientations, get_skin=True)

                # only take the 23 joint orientations, not the global orientation
                generated_pred_Rs = generated_pred_Rs[:,1:,:]

                fake_joints.append(generated_joints)
                all_fake_Rs.append(generated_pred_Rs)

                # for visualization
                all_pred_verts.append(tf.gather(generated_verts, self.show_these))
                all_pred_cams.append(tf.gather(generated_cams, self.show_these))

                ##############################################################################################
                # Calculate Generator Losses
                ##############################################################################################

                # calculate keypoint reprojection loss
                pred_kp = batch_orth_proj_idrot(generated_joints, generated_cams,
                                                    name='proj2d_stage%d' % i)

                # For visualization
                all_pred_kps.append(tf.gather(pred_kp, self.show_these))
                kpr_losses.append(self.kpr_loss_weight * joint_reprojection_loss(kp2d_gts, pred_kp))

                # calculate mesh reprojection loss
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

                # calculate ctritic loss
                if not self.encoder_only:

                    # get kcs matrix
                    kcs = get_kcs(generated_joints, self.C)

                    # forward pass through critic network for generator update
                    generator_critic_out = self.critic_network([kcs,
                                                       generated_joints[:,:self.num_joints,:],
                                                       generated_shapes,
                                                       generated_pred_Rs],
                                                       training=False
                                                     )

                    # compute generator loss caused by critic
                    generator_critic_loss = - tf.reduce_sum(tf.reduce_mean(generator_critic_out, 0))
                    generator_critic_losses.append(generator_critic_loss * self.critic_loss_weight)

                    if self.debug:
                        tf.print("generator critic loss", generator_critic_loss)

                # Finally update to end iteration.
                theta_prev = theta_here

            ###########################################################################################
            # Generator optimization
            ###########################################################################################

            # set variables to be trained
            variables = self.image_feature_extractor.trainable_variables + self.generator3d.trainable_variables
            variables.append(self.mean_var)

            if self.debug:
                tf.print("variables len", len(variables))

            # compute overall generator loss
            generator_loss_sum = 0.
            if self.use_kpr_loss:
                #generator_loss_sum.append(kpr_losses[-1])
                generator_loss_sum += kpr_losses[-1]
            if self.use_mesh_repro_loss:
                generator_loss_sum += mr_losses[-1]
            if not self.encoder_only:
                generator_loss_sum += generator_critic_losses[-1]
                pass

            if self.debug:
                tf.print("gen loss sum", generator_loss_sum)

            # get gradients
            gradients_of_generator = gen_tape.gradient(generator_loss_sum, variables)

            # optimize generator
            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, variables))


        #############################################################################################
        # Critic update
        #############################################################################################
        all_fake_joints = tf.concat(fake_joints, 0)[:,:self.num_joints,:]

        if not self.encoder_only:
            with tf.GradientTape() as critic_tape:
                all_fake_Rs = tf.concat(all_fake_Rs, 0)
                all_fake_shapes = tf.concat(fake_shapes, 0)
                joint3d_gt = joint3d_gt[:,:self.num_joints,:]

                if self.debug:
                    tf.print('real_joints', joint3d_gt)
                    tf.print('real_shapes', shape_gts)
                    tf.print('fake_joints', fake_joints)
                    tf.print('fake_shapes', fake_shapes)
                    tf.print('all_fake_joints', all_fake_joints)
                    tf.print('all_fake_shapes', all_fake_shapes)

                real_kcs = get_kcs(joint3d_gt, self.C)

                # forward pass with real data
                real_output = self.critic_network([real_kcs,
                                                   joint3d_gt,
                                                   shape_gts,
                                                   Rs_gt],
                                                   training=True)

                fake_kcs = get_kcs(all_fake_joints, self.C)

                # forward pass with fake data
                fake_output = self.critic_network([fake_kcs,
                                                   all_fake_joints,
                                                   all_fake_shapes,
                                                   all_fake_Rs],
                                                   training=True)

                # WGAN loss
                critic_network_loss = tf.reduce_sum(tf.reduce_mean(fake_output - real_output, 0))

                # compute the gradient penalty for the improved WGAN loss
                if self.use_gradient_penalty:
                    # interpolated inputs for the discriminator forward pass
                    alpha = tf.random.uniform(all_fake_joints.shape)
                    beta = tf.random.uniform(all_fake_shapes.shape)
                    gamma = tf.random.uniform(all_fake_Rs.shape)
                    interpolated_joints = all_fake_joints + alpha * (joint3d_gt - all_fake_joints)
                    interpolated_kcs = get_kcs(interpolated_joints, self.C)
                    interpolated_shapes = all_fake_shapes + beta * (shape_gts - all_fake_shapes)
                    interpolated_Rs = all_fake_Rs + gamma * (Rs_gt - all_fake_Rs)

                    # forward pass with interpolated inputs
                    out_interpolated = self.critic_network([interpolated_kcs,
                                                            interpolated_joints[:, :self.num_joints, :],
                                                            interpolated_shapes,
                                                            interpolated_Rs],
                                                           training=False)
                    # get gradients
                    gradients_to_penalize = tf.gradients(ys=out_interpolated,
                                                         xs=[interpolated_kcs,
                                                             interpolated_joints,
                                                             interpolated_shapes,
                                                             interpolated_Rs])

                    penalty = compute_gradient_penalty(gradients_to_penalize, self.debug)

                    # use 10 for the gradient penalty weight as in the improved WGAN paper
                    critic_network_loss = critic_network_loss + 10. * penalty

            # get gradients of discriminator
            gradients_of_discriminator = critic_tape.gradient(critic_network_loss,
                                                            self.critic_network.trainable_variables)

            # optimize critic network
            self.critic_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                        self.critic_network.trainable_variables))

            if self.debug:
                tf.print("variables critic network length:", len(self.critic_network.trainable_variables))
                tf.print("gradients critic length:", len(gradients_of_discriminator))
                tf.print("critic network loss:", critic_network_loss)

        #################################################################################################
        # Return results
        #################################################################################################
        result = {}

        result["kpr_losses"] = kpr_losses
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

        # seg gt needs to be same dimension as color image.
        seg_gt = seg_gt.squeeze()
        seg2_gt = np.stack((seg_gt,seg_gt,seg_gt), axis=2)
        rend_seg_gt = renderer(vert + cam_t, cam_for_render, img=seg2_gt)
        combined = np.hstack([skel_img, rend_img / 255., rend_seg_gt / 255.])
        return combined


    def draw_results(self, imgs, segs_gt, gt_kps, est_verts, pred_kps, cam, step):
        import io
        import matplotlib.pyplot as plt

        imgs = imgs.numpy()
        segs_gt = segs_gt.numpy()
        gt_kps = gt_kps.numpy()
        est_verts = est_verts.numpy()
        pred_kps = pred_kps.numpy()
        cam = cam.numpy()

        if self.data_format == 'NCHW':
            imgs = np.transpose(imgs, [0, 2, 3, 1])

        for img_id, (img, seg_gt, gt_kp, verts, keypoints, cams) in enumerate(
                zip(imgs, segs_gt, gt_kps, est_verts, pred_kps, cam)):
            # verts, joints, cams are a list of len T.
            all_rend_imgs = []
            for vert, joint, cam in zip(verts, keypoints, cams):

                rend_img = self.visualize_img(img, gt_kp, vert, joint, cam,
                                                  self.renderer, seg_gt)
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

    '''
    This method trains the network as specified when initializing the Trainer
    '''
    def train(self):
        print('started training')
        print('...')

        # Initialize variables
        self.mean_var = self.load_mean_param()
        self.global_step = tf.Variable(1, name='global_step', trainable=False, dtype=tf.int64)
        step = tf.Variable(0, name='step', trainable=False, dtype=tf.int64)

        itr = 0
        epoch = 0

        # Restore checkpoint from checkpoint_dir
        if self.train_from_checkpoint:
            print("restore checkpoint")
            self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
            print("checkpoint restored")

        # Start looping through dataset, since repeat() is called on them, they repeat infinitely
        # until break is called when max_epochs is reached.
        for data_gen, data_critic, val_data in self.full_dataset:
            if itr == 0:
                start_time = time.time()

            # Get data from data loaders
            images, segmentations, keypoints = data_gen
            if not self.encoder_only or self.do_bone_evaluation:
                joints, shapes, rotations = data_critic
                joints = tf.squeeze(joints, axis=1)
                rotations = tf.squeeze(rotations)[:,1:,:]
            else:
                joints, shapes, rotations = None, None, None

            # Do an update step
            result = self.train_step(images, segmentations, keypoints, joints, shapes,
                                     rotations, step)

            step = step + 1

            ############################################################################################
            # Write Generator update to tensorboard
            ############################################################################################

            with self.training_writer.as_default():
                # make sure only necessary information is printed to tensorboard
                if self.use_kp_loss:
                    tf.summary.scalar('generator/kp_loss', result["kp_losses"][-1], step=step)
                if self.use_mesh_repro_loss:
                    tf.summary.scalar('generator/mr_loss', result["mr_losses"][-1], step=step)
                if self.do_bone_evaluation:
                    tf.summary.scalar('bones/avg_total_bone_lenth_pred', result["avg_total_bone_length_pred"], step=step)
                    tf.summary.scalar('bones/avg_total_bone_lenth_gt', result["avg_total_bone_length_gt"], step=step)

                # every log_img_step draw images and write them to tensorboard
                if tf.equal((step % self.log_img_step), tf.constant(0, dtype=tf.int64)):
                    show_images = tf.gather(images, self.show_these)
                    show_segs = tf.gather(segmentations, self.show_these)
                    show_kps = tf.gather(keypoints, self.show_these)
                    print("drawing images now")
                    self.draw_results(show_images, show_segs, show_kps,
                                       result["generated_verts"], result["generated_kps"],
                                       result["generated_cams"], step)

                ############################################################################################
                # Write Critic update to tensorboard
                ############################################################################################
                if not self.encoder_only:
                    tf.summary.scalar('critic/critic_network_loss', result["critic_network_loss"], step=step)
                    tf.summary.scalar('critic/generator_critic_loss',
                                      result["generator_critic_losses"][-1], step=step)
                    tf.summary.scalar('critic/penalty', result["critic_penalty"], step=step)
                self.training_writer.flush()

            ################################################################################################
            # Validation step
            ################################################################################################
            # If validation should be performed ervery val_step_size validate the current model on
            # a single batch and print results to tensorboard, if it is also at log_img_step then
            # draw validation images as well
            if self.use_validation and tf.equal((step % self.val_step_size), tf.constant(0, dtype=tf.int64)):
                # Get data
                val_images, val_segmentations, val_keypoints = val_data

                # Perform validation step
                val_result = self.val_step(val_images, val_segmentations, val_keypoints)

                print("VALIDATION!")
                print("validation: kp:", result["kp_losses"][-1].numpy())#, ", mr:", result["mr_losses"][-1].numpy())#, "critic:", result["generator_critic_losses"][-1].numpy())
                with self.val_writer.as_default():
                    tf.summary.scalar('generator/kp_loss', val_result["kp_losses"][-1], step=step)
                    if self.use_mesh_repro_loss:
                        tf.summary.scalar('generator/mr_loss', val_result["mr_losses"][-1], step=step)

                    if tf.equal((step % self.log_img_step), tf.constant(0, dtype=tf.int64)):
                        show_images = tf.gather(val_images, self.show_these)
                        show_segs = tf.gather(val_segmentations, self.show_these)
                        show_kps = tf.gather(val_keypoints, self.show_these)
                        print("drawing images now")
                        self.draw_results(show_images, show_segs, show_kps,
                                          val_result["generated_verts"], val_result["generated_kps"],
                                          val_result["generated_cams"], step)
                    self.val_writer.flush()

            itr += 1
            print("itr", itr, "/", self.num_itr_per_epoch, ", epoch:", epoch, "in", (end_time - start_time), "s")
            if not self.encoder_only:
                print("critic_loss", result['critic_network_loss'].numpy(), ", penalty:", result["critic_penalty"].numpy())
            print("generator: kp:", result["kp_losses"][-1].numpy())#, ", mr:", result["mr_losses"][-1].numpy())#, "critic:", result["generator_critic_losses"][-1].numpy())
            if itr >= self.num_itr_per_epoch:
                end_time = time.time()
                itr = 0
                epoch += 1

                if epoch % 5 == 0:
                    self.checkpoint.save(self.checkpoint_prefix)

                if epoch >= self.max_epoch:
                    break
                print("Starting epoch %i, approx finished in %f4 min. Training finished at approx %i:%i", )

                print("epoch", epoch)
        print('Finish training on %s' % self.model_dir)


    def validate_checkpoint(self,draw_best_worst = False, draw_every_image = False):
        self.mean_var = self.load_mean_param()
        print(self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir)).expect_partial())

        kp_losses = []
        mr_losses = []
        step = 0
        if draw_best_worst:
            best_kp = {"val": 10}
            best_mr = {"val": 100}
            worst_kp = {"val": 0}
            worst_mr = {"val": 0}
            best_combined = {"val": 100}
            worst_combined = {"val": 0}

        writer = tf.summary.create_file_writer(join(self.logs, self.checkpoint_dir))

        for images, segmentations, keypoints in self.val_dataset:
            step += 1
            print("step", step)
            val_result = self.val_step(images, segmentations, keypoints)
            kp_losses.append(val_result["kp_losses"][-1])
            mr_losses.append(val_result["mr_losses"][-1])

            if draw_every_image:
                with writer.as_default():
                    self.draw_results(images, segmentations, keypoints, val_result["generated_verts"],
                                      val_result["generated_kps"],val_result["generated_cams"],step)
                    writer.flush()

            if draw_best_worst:
                if best_kp["val"] >= kp_losses[-1]:
                    best_kp.update({
                        "val": kp_losses[-1],
                        "images": images,
                        "segmentations": segmentations,
                        "keypoints": keypoints,
                        "verts": val_result["generated_verts"],
                        "kps": val_result["generated_kps"],
                        "cams": val_result["generated_cams"]
                    })

                if worst_kp["val"] <= kp_losses[-1]:
                    worst_kp.update({
                        "val": kp_losses[-1],
                        "images": images,
                        "segmentations": segmentations,
                        "keypoints": keypoints,
                        "verts": val_result["generated_verts"],
                        "kps": val_result["generated_kps"],
                        "cams": val_result["generated_cams"]
                    })

                if best_mr["val"] >= mr_losses[-1]:
                    best_mr.update({
                        "val": mr_losses[-1],
                        "images": images,
                        "segmentations": segmentations,
                        "keypoints": keypoints,
                        "verts": val_result["generated_verts"],
                        "kps": val_result["generated_kps"],
                        "cams": val_result["generated_cams"]
                    })
                if worst_mr["val"] <= mr_losses[-1]:
                    worst_mr.update({
                        "val": mr_losses[-1],
                        "images": images,
                        "segmentations": segmentations,
                        "keypoints": keypoints,
                        "verts": val_result["generated_verts"],
                        "kps": val_result["generated_kps"],
                        "cams": val_result["generated_cams"]
                    })

                if best_combined["val"] >= (mr_losses[-1] + kp_losses[-1]):
                    best_combined.update({
                        "val": (mr_losses[-1] + kp_losses[-1]),
                        "images": images,
                        "segmentations": segmentations,
                        "keypoints": keypoints,
                        "verts": val_result["generated_verts"],
                        "kps": val_result["generated_kps"],
                        "cams": val_result["generated_cams"]
                    })
                if worst_combined["val"] <= (mr_losses[-1] + kp_losses[-1]):
                    worst_combined.update({
                        "val": (mr_losses[-1] + kp_losses[-1]),
                        "images": images,
                        "segmentations": segmentations,
                        "keypoints": keypoints,
                        "verts": val_result["generated_verts"],
                        "kps": val_result["generated_kps"],
                        "cams": val_result["generated_cams"]
                    })

        if draw_best_worst:
            result_dicts = [best_kp, worst_kp, best_mr, worst_mr, best_combined, worst_combined]
            with writer.as_default():
                step = 0
                for d in result_dicts:
                    self.draw_results(d["images"], d["segmentations"], d["keypoints"], d["verts"], d["kps"], d["cams"], step)
                    writer.flush()
                    step += 1
        print(kp_losses)
        print(mr_losses)
        print("average kp_loss =", np.mean(kp_losses))
        print("average mr_loss =", np.mean(mr_losses))
