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

from time import time
import tensorflow as tf
import numpy as np

from tensorflow.python.ops import resources
from tensorflow.python.ops import variables

from os.path import join, dirname
import deepdish as dd

# For drawing
from .util import renderer as vis_util
#from .util.data_utils import get_silhouette_from_seg_im as get_sil

class HMRTrainer(object):
    def __init__(self, config, dataset, mocap_dataset = None):
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

        # Data size
        self.img_size = config.img_size
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
        self.generator_writer = tf.summary.create_file_writer(self.model_dir+'generator')
        self.critic_writer = tf.summary.create_file_writer(self.model_dir+'critic')

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
            self.full_dataset.append(tf.data.Dataset.range(1))
        # create one dataset so its easier to iterate over it
        self.full_dataset = tf.data.Dataset.zip(tuple(self.full_dataset))

        #######################################################################################
        # Set up losses, optimizers and models
        #######################################################################################

        self.keypoint_loss = keypoint_l1_loss
        self.mesh_repro_loss = mesh_reprojection_loss


        # Initialize optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(self.generator_lr)
        self.critic_optimizer = tf.keras.optimizers.RMSprop(self.critic_lr)

        # Load models
        self.image_feature_extractor = Encoder_resnet()
        self.generator3d = Encoder_fc3_dropout()
        self.critic_network = Critic_network()

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
        self.mean_var = tf.Variable(
            mean, name="mean_param", dtype=tf.float32, trainable=True)
        init_mean = tf.tile(self.mean_var, [self.batch_size, 1])
        return init_mean

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(self, images, seg_gts, kp2d_gts, joint3d_gt, shape_gts, step):

        #############################################################################################
        # Generator update
        #############################################################################################

        batched_images = tf.split(images, self.num_gen_steps_per_itr)
        batched_seg_gts = tf.split(seg_gts, self.num_gen_steps_per_itr)
        batched_kp2d_gts = tf.split(kp2d_gts, self.num_gen_steps_per_itr)

        #Do more generator steps than critic steps per train
        for small_images, small_seg, small_kps in zip(batched_images,
                                                      batched_seg_gts,
                                                      batched_kp2d_gts):
            # First make sure data_format is right
            if self.data_format == 'NCHW':
                # B x H x W x 3 --> B x 3 x H x W
                small_images = tf.transpose(small_images, [0, 3, 1, 2])
                small_seg = tf.transpose(small_seg, [0, 3, 1, 2])

            fake_joints = []
            kp_losses = []
            mr_losses = []
            cr_losses = []

            with tf.GradientTape() as gen_tape:
                #Extract feature vector from image using resnet
                extracted_features = self.image_feature_extractor(small_images, training=False)

                theta_prev = self.load_mean_param()
                # Main IEF loop
                for i in np.arange(self.num_stage):
                    print('iteration', i)
                    state = tf.concat([extracted_features, theta_prev], 1)
                    print(state.shape)
                    #TODO how do i get reuse=true for this?
                    if i != 2:
                        delta_theta = self.generator3d(state, training=False)
                    else:
                        delta_theta = self.generator3d(state, training=True)#, reuse=True)

                    # Compute new theta
                    theta_here = theta_prev + delta_theta
                    # cam = N x 3, pose N x self.num_theta, shape: N x 10
                    generated_cams = theta_here[:, :self.num_cam]
                    generated_poses = theta_here[:, self.num_cam:(self.num_cam + self.num_theta)]
                    generated_shapes = theta_here[:, (self.num_cam + self.num_theta):]
                    # Rs_wglobal is Nx24x3x3 rotation matrices of poses
                    generated_verts, generated_joints, generated_pred_Rs = self.smpl(generated_shapes, generated_poses, get_skin=True)
                    fake_joints.append(generated_joints)

                    #Calculate keypoint reprojection
                    if self.use_kp_loss:
                        pred_kp = batch_orth_proj_idrot(generated_joints, generated_cams, name='proj2d_stage%d' % i)


                    #Calculate mesh reprojection
                    if self.use_mesh_repro_loss:
                        pred_silhouette = reproject_vertices(generated_verts,
                                                             generated_cams,
                                                             curr_images.shape[1:3],
                                                             name='mesh_reproject_stage%d' % i)

                    ##############################################################################################
                    # Calculate Generator Losses
                    ##############################################################################################

                    #Calculate keypoint reprojection loss
                    if not self.use_mesh_repro_loss:
                        kp_losses.append(
                            self.generator_kp_loss_weight * self.keypoint_loss(small_kps, pred_kp)
                        )

                    #Calculate mesh reprojection loss
                    if self.use_mesh_repro_loss:
                        # silhouette_gt: first entry = index sample; 
                        #                second,third = coordinate of pixel with value > 0.
                        silhouette_gt = tf.where(tf.greater(small_seg, 0.))[:, :3]
                        self.silhouette_pred = silhouette_pred
                        self.silhouette_gt = small_seg

                        repro_loss = self.mesh_repro_loss(
                            silhouette_gt, silhouette_pred, self.batch_size, name='mesh_repro_loss' % i)
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
                        kcs = get_kcs(generated_joints, self.C)
                        generated_joints = tf.transpose(generated_joints, perm=[0,2,1])[:,:,:14]
                        critic_loss = self.critic_network([kcs,
                                                           generated_joints,
                                                           generated_shapes],
                                                           training=True
                                                         )
                        cr_losses.append(critic_loss *
                                             self.critic_loss_weight)


                    # Save things for visualiations:
                    #self.all_verts.append(tf.gather(verts, self.show_these))
                    #if(not self.use_mesh_repro_loss):
                    #self.all_pred_kps.append(tf.gather(pred_kp, self.show_these))
                    #if(self.use_mesh_repro_loss):
                    #    self.all_pred_silhouettes.append(tf.gather(silhouette_pred, self.show_these))
                    #self.all_pred_cams.append(tf.gather(cams, self.show_these))

                    # Finally update to end iteration.
                    theta_prev = theta_here

            ###########################################################################################
            # Generator optimization
            ###########################################################################################

            # Calculate overall generator loss
            generator_loss = 0
            # Just the last loss.
            if self.use_kp_loss:
                generator_loss += kp_losses[-1]
                print("generator_loss", generator_loss)

            if self.use_mesh_repro_loss:
                generator_loss += mr_losses[-1]
                print("generator_loss", generator_loss)

            if not self.encoder_only:
                generator_loss += (-self.critic_loss_weight * critic_loss)
                print("critic loss", critic_loss)

            ##TODO no 3d labels used yet
            #if self.use_3d_label:
            #    self.generator_loss_3d = loss_3d_params[-1]
            #    self.generator_loss_3d_joints = loss_3d_joints[-1]

            #  self.generator_loss += (self.generator_loss_3d + self.generator_loss_3d_joints)

            #all_train_vars = self.generator3d.trainable_variables + self.image_feature_extractor.trainable_variables

            print("generator_loss", generator_loss)
            print("gen_tape", gen_tape)
            print('trainable_variables', self.generator3d.trainable_variables)
            gradients_of_generator = gen_tape.gradient(generator_loss,
                                                       self.generator3d.trainable_variables)
            print("gradients", gradients_of_generator)
            self.generator_optimizer.apply_gradients(zip(gradients_of_generator,
                                                    self.generator3d.trainable_variables))

            print("step %g: time %g, generator_loss: %.4f" %(step, 0, generator_loss))
            step += 1

        ############################################################################################
        # Write Generator update to tensorboard
        ############################################################################################
        with self.generator_writer.as_default():
            self.generator_writer.scalar('total_loss', generator_loss,
                                         step=step)
            self.generator_writer.scalar('kp_loss', kp_losses[-1], step=step)
            self.generator_writer.scalar('mr_loss', mr_losses[-1], step=step)

            if step % self.log_img_step == 0:
                print('i would be drawing images now')
                self.draw_results(small_images, small_seg, small_kps, est_verts, joints, cam, step,
                                 self.generator_writer)
                #self.draw_results(result, self.generator_writer)

        #############################################################################################
        # Critic update
        #############################################################################################

        if not self.encoder_only:
            with tf.GradientTape as critic_tape:
                all_fake_joints = tf.concat(fake_joints, 0)
                print('real_joints', joint3d_gt)
                print('fake_joints', all_fake_joints)
                all_fake_shapes = tf.concat(fake_shapes, 0)

                real_kcs = get_kcs(joint3d_gt, self.C)
                real_output = self.critic_network([joint3d_gt, shape_gts,
                                                  real_kcs], training=True)

                fake_kcs = get_kcs(all_fake_joints, self.C)
                fake_output = self.critic_network([all_fake_joints,
                                                  all_fake_shapes, fake_kcs],
                                                  training=True)

                critic_loss = -(real_output - fake_output)

            gradients_of_discriminator = disc_tape.gradient(critic_loss,
                                                            self.critic_network.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                        self.critic_network.trainable_variables))

            print("step %g: time %g, generator_loss: %.4f" %(step, 0, critic_loss))

            ############################################################################################
            # Write Critic update to tensorboard
            ############################################################################################
            with self.critic_writer.as_default():
                self.critic_writer.scalar('critic_loss', critic_loss,
                                             step=step)



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

    def draw_results(self, imgs, segs_gt, gt_kps, est_verts, joints, cam, step, writer):
        import io
        import matplotlib.pyplot as plt
        import cv2

        if self.data_format == 'NCHW':
            imgs = np.transpose(imgs, [0, 2, 3, 1])

        img_summaries = []

        if not self.use_mesh_repro_loss:
            segs_gt = np.empty_like(imgs)

        for img_id, (img, seg_gt, gt_kp, verts, joints, cams) in enumerate(
                zip(imgs, segs_gt, gt_kps, est_verts, joints, cams)):
            # verts, joints, cams are a list of len T.
            all_rend_imgs = []
            for vert, joint, cam in zip(verts, joints, cams):

                if(self.use_mesh_repro_loss):
                    rend_img = self.visualize_img(img, gt_kp, vert, joint, cam,
                                                  self.renderer, seg_gt)
                else:
                    rend_img = self.visualize_img(img, gt_kp, vert, joint, cam,
                                                  self.renderer)
                all_rend_imgs.append(rend_img)
            combined = np.vstack(all_rend_imgs)

            sio = io.BytesIO()

            plt.imsave(sio, combined, format='png')
            sio.seek(0)

            with writer.as_default():
                print("img_summaries.append")
                writer.image(("vis_images/%d" % img_id), vis_sum, step=step)

            sio.flush()
            sio.close()
            plt.close()

    def train(self):
        print('started training')
        # For rendering!

        print('...')
        for epoch in range(self.max_epoch):

            itr = 0
            for data_gen, data_critic in self.full_dataset:

                images, segmentations, keypoints = data_gen
                if not self.encoder_only:
                    joints, shapes = data_critic
                else:
                    joints = None
                    shapes = None

                self.train_step(images, segmentations, keypoints, joints,
                                shapes, epoch*self.num_itr_per_epoch + itr)

                itr += self.num_gen_steps_per_itr
                if itr >= self.num_itr_per_epoch:
                    break


        print('Finish training on %s' % self.model_dir)
