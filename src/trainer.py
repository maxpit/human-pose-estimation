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
from .models import Critic_network, get_encoder_fn_separate,
precompute_C_matrix, get_kcs

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
        # Config + path
        self.config = config

        self.model_dir = config.model_dir
        print('model dir: %s', self.model_dir)
        self.load_path = config.load_path
        print('load path: %s', self.load_path)

        self.data_format = config.data_format
        print('data_format: %s', self.data_format)
        self.smpl_model_path = config.smpl_model_path
        print('smpl_model_path: %s', self.smpl_model_path)
        self.pretrained_model_path = config.pretrained_model_path
        print('pretrained_model_path: %s', self.pretrained_model_path)
        self.encoder_only = config.encoder_only
        print('encoder only:', self.encoder_only)
        self.use_3d_label = config.use_3d_label
        print('use_3d_label:', self.use_3d_label)

        # Data size
        self.img_size = config.img_size
        print('image_size:', self.img_size)
        self.num_stage = config.num_stage
        print('num_stage:', self.num_stage)
        self.batch_size = config.batch_size
        print('batch_size:', self.batch_size)
        self.max_epoch = config.epoch

        self.num_cam = 3
        self.proj_fn = batch_orth_proj_idrot

        self.num_theta = 72  # 24 * 3
        self.total_params = self.num_theta + self.num_cam + 10
        self.use_mesh_repro_loss = config.use_mesh_repro_loss
        self.use_kp_loss= config.use_kp_loss

        # Data
        num_images = num_examples(config.datasets)
        print('num_images: ', num_images)
        num_mocap = num_examples(config.mocap_datasets)
        print('num_mocap', num_mocap)

        self.num_itr_per_epoch = num_images / self.batch_size
        self.num_mocap_itr_per_epoch = num_mocap / self.batch_size

        self.val_step = config.validation_step_size

        #####################################################################
        # OUR CODE
        #####################################################################

        ## Build datasets

        # Create train and validation dataset from dataset with given train/val
        # split

        self.full_dataset = []

        if config.use_validation is not 0.0:
            self.use_val = True

            num_train_samples = int(num_images * config.train_val_split)

            print("NUM_TRAIN_SAMPLES:",num_train_samples)
            train_set = dataset.take(num_train_samples).shuffle(buffer_size=10000).repeat()
            val_set = dataset.skip(num_train_samples).shuffle(buffer_size=10000).repeat()

            train_set = train_set.batch(self.batch_size)
            val_set = val_set.batch(self.batch_size)

            self.full_dataset.append(train_set)
        else:
            self.use_val = False
            dataset = dataset.shuffle(buffer_size=10000).repeat()
            dataset = dataset.batch(self.batch_size)
            self.full_dataset.append(dataset)

        # Formats
        # image: B x H x W x 3
        # seg_gt: B x H x W x 1
        # kp_gt: B x 19 x 3
        self.image, self.seg_gt, self.kp_gt = self.iterator.get_next()

        if not self.encoder_only:
            critic_dataset = mocap_dataset.shuffle(buffer_size=10000).repeat()
            critic_dataset = critic_dataset.batch(self.batch_size*3)
            self.full_dataset.append(critic_dataset)

        self.full_dataset = tf.data.Dataset.zip(self.full_dataset)

        #####################################################################
        # END OF OUR CODE
        #####################################################################

        self.a = tf.reshape(self.kp_gt, (-1,3))

        #TODO put this in train_set
        # First make sure data_format is right
        if self.data_format == 'NCHW':
            # B x H x W x 3 --> B x 3 x H x W
            self.image = tf.transpose(self.image, [0, 3, 1, 2])
            self.seg_gt = tf.transpose(self.seg_gt, [0, 3, 1, 2])

#        if self.use_3d_label:
#            self.poseshape_loader = data_loader['label3d']
#            # image_loader[3] is N x 2, first column is 3D_joints gt existence,
#            # second column is 3D_smpl gt existence
#            self.has_gt3d_joints = data_loader['has3d'][:, 0]
#            self.has_gt3d_smpl = data_loader['has3d'][:, 1]


        self.log_img_step = config.log_img_step

        # For visualization:
        num2show = np.minimum(6, self.batch_size)
        # Take half from front & back
        self.show_these = tf.constant(
            np.hstack(
                [np.arange(num2show / 2), self.batch_size - np.arange(3) - 1]),
            tf.int32)


        self.validation_step_size = config.validation_step_size
        # Model spec
        self.model_type = config.model_type
        self.keypoint_loss = keypoint_l1_loss
        self.mesh_repro_loss = mesh_reprojection_loss

        # Optimizer, learning rate
        self.generator_lr = config.generator_lr
        self.critic_lr = config.critic_lr
        # Weight decay
        self.e_wd = config.e_wd
        self.d_wd = config.d_wd

        self.generator_kp_loss_weight = config.generator_loss_weight
        self.critic_loss_weight = config.critic_loss_weight
        self.e_3d_weight = config.e_3d_weight

        self.mr_loss_weight = config.mr_loss_weight

        self.generator_optimizer = tf.keras.optimizers.Adam(self.generator_lr)
        self.critic_optimizer = tf.keras.optimizers.RMSprop(self.critic_lr)
        # Instantiate SMPL
        self.smpl = SMPL(self.smpl_model_path)
        self.E_var = []

        self.C = precompute_C_matrix()

        #TODO probably bin this
        # Logging
        init_fn = None
        if self.use_pretrained():
            # Make custom init_fn
            print("Fine-tuning from %s" % self.pretrained_model_path)
            if 'resnet_v2_50' in self.pretrained_model_path:
                resnet_vars = [
                    var for var in self.E_var if 'resnet_v2_50' in var.name
                ]
                self.pre_train_saver = tf.train.Saver(resnet_vars)
            elif 'pose-tensorflow' in self.pretrained_model_path:
                resnet_vars = [
                    var for var in self.E_var if 'resnet_v1_101' in var.name
                ]
                self.pre_train_saver = tf.train.Saver(resnet_vars)
            else:
                self.pre_train_saver = tf.train.Saver()

            def load_pretrain(sess):
                self.pre_train_saver.restore(sess, self.pretrained_model_path)

            init_fn = load_pretrain

        self.summery_writer_train = tf.summary.FileWriter(self.model_dir+'train')
        self.summary_writer_critic = tf.summary.FileWriter(self.model_dir+'critic')
        self.summary_writer_val = tf.summary.FileWriter(self.model_dir+'val')
        
        ##############################################################################
        # Load models
        ##############################################################################

        self.image_feature_extractor, self.generator3d = get_encoder_fn_separate(self.model_type)
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
        self.E_var.append(self.mean_var)
        init_mean = tf.tile(self.mean_var, [self.batch_size, 1])
        return init_mean

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(images, seg_gts, kp2d_gts, pose_gts, shape_gts):
        noise = tf.random.normal([BATCH_SIZE, noise_dim])

        #Extract feature vector from image using resnet
        extracted_features = self.image_feature_extractor(images, training=True)

        #############################################################################################
        # Generator update
        #############################################################################################

        #Do more generator steps than critic steps per train
        for steps in range(self.num_gen_steps_per_itr):
            theta_prev = self.load_mean_param()

            kp_losses = []
            mr_losses = []
            cr_losses = []

            with tf.GradientTape() as gen_tape:
                # Main IEF loop
                for i in np.arange(self.num_stage):
                    state = tf.concat([extracted_features, theta_prev], 1)

                    #TODO how do i get reuse=true for this?
                    if i == 0:
                        delta_theta = self.generator3d(state, training=True)
                    else:
                        delta_theta = self.generator3d(state, training=True, reuse=True)

                    # Compute new theta
                    theta_here = theta_prev + delta_theta
                    # cam = N x 3, pose N x self.num_theta, shape: N x 10
                    generated_cams = theta_here[:, :self.num_cam]
                    generated_poses = theta_here[:, self.num_cam:(self.num_cam + self.num_theta)]
                    generated_shapes = theta_here[:, (self.num_cam + self.num_theta):]
                    # Rs_wglobal is Nx24x3x3 rotation matrices of poses
                    generated_verts, generated_joints, generated_pred_Rs = self.smpl(generated_shapes, generated_poses, get_skin=True)

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
                            self.generator_kp_loss_weight * self.keypoint_loss(self.kp_gt, pred_kp)
                        )

                    #Calculate mesh reprojection loss
                    if self.use_mesh_repro_loss:
                        # silhouette_gt: first entry = index sample; 
                        #                second,third = coordinate of pixel with value > 0.
                        silhouette_gt = tf.where(tf.greater(curr_seg_gt, 0.))[:, :3]
                        self.silhouette_pred = silhouette_pred
                        self.silhouette_gt = silhouette_gt

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
                        critic_loss = self.critic_network([kcs,
                                                           generated_joints,
                                                           generated_shapes],
                                                           training=True
                                                         )
                        cr_losses.append(critic_loss *
                                             self.critic_loss_weight)


                    # Save things for visualiations:
                    self.all_verts.append(tf.gather(verts, self.show_these))
                    #if(not self.use_mesh_repro_loss):
                    self.all_pred_kps.append(tf.gather(pred_kp, self.show_these))
                    if(self.use_mesh_repro_loss):
                        self.all_pred_silhouettes.append(tf.gather(silhouette_pred, self.show_these))
                    self.all_pred_cams.append(tf.gather(cams, self.show_these))

                    # Finally update to end iteration.
                    theta_prev = theta_here

            ###########################################################################################
            # Generator optimization
            ###########################################################################################

            # Calculate overall generator loss
            generator_loss = 0
            # Just the last loss.
            if self.use_kp_loss:
                generator_loss += loss_kps[-1]

            if self.use_mesh_repro_loss:
                generator_loss += loss_mr[-1]

            if not self.encoder_only:
                generator_loss += (-self.critic_loss_weight * self.critic_loss_fake)\
                                         + self.generator_loss_kp\
                                         + self.generator_loss_mr

#                #TODO no 3d labels used yet
#                if self.use_3d_label:
#                    self.generator_loss_3d = loss_3d_params[-1]
#                    self.generator_loss_3d_joints = loss_3d_joints[-1]
#
#                    self.generator_loss += (self.generator_loss_3d + self.generator_loss_3d_joints)

            all_train_vars = self.generator3d.trainable_variables + self.image_feature_extractor.trainable_variables

            gradients_of_generator = gen_tape.gradient(generator_loss, all_train_vars)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, all_train_vars))

        #############################################################################################
        # Critic update
        #############################################################################################

        if not self.encoder_only:
            with tf.GradientTape as critic_tape:
                all_fake_joints = tf.concat(fake_joints, 0)

                print('real_joints', self.joints)
                print('fake_joints', all_fake_joints)

                combined_joints = tf.concat([tf.squeeze(self.joints), all_fake_joints], 0,
                                       name="combined_joints")

                all_fake_shapes = tf.concat(fake_shapes, 0)
                combined_shapes = tf.concat([self.real_shapes, all_fake_shapes], 0, name="combined_shape")

                disc_input = {
                    'weight_decay': self.d_wd,
                    'shapes': combined_shapes,
                    'joints': combined_joints
                }

                self.d_out, self.D_var = Critic_network(**disc_input, C_matrix=self.C,
                                                        batch_size=self.batch_size)
                self.d_out_real, self.d_out_fake = tf.split(self.d_out, 2)

                # Compute losses:
                with tf.name_scope("comp_critic_loss"):
                    self.critic_loss_fake = losses.compute_weighted_loss(self.d_out_fake)
                    self.critic_loss_real = losses.compute_weighted_loss(self.d_out_real)
                    if not self.encoder_only:
                        with tf.name_scope("gather_critic_loss"):
                            self.critic_loss = self.critic_loss_weight * (
                                (self.critic_loss_fake - self.critic_loss_real)
                            )
                                #* self.c_lamda *  )


            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))



    def train_step(self):
        # Extract image features.

        self.img_feat, self.E_var = img_enc_fn(
            self.image, weight_decay=self.e_wd, reuse=False)

        loss_mr = []
        loss_kps = []

        if self.use_3d_label:
            loss_3d_joints, loss_3d_params = [], []
        # For discriminator
        fake_joints, fake_shapes = [], []
        # Start loop
        # 85D
        theta_prev = self.load_mean_param()

        # For visualizations
        self.all_verts = []
        self.all_pred_kps = []
        self.all_pred_silhouettes = []
        self.all_pred_cams = []
        self.all_delta_thetas = []
        self.all_theta_prev = []

        # Main IEF loop
        for i in np.arange(self.num_stage):
            print('Iteration %d' % i)
            # ---- Compute outputs
            print("img_feat: ", self.img_feat)

            print("theta_prev: ", theta_prev)
            state = tf.concat([self.img_feat, theta_prev], 1)

            if i == 0:
                delta_theta, threeD_var = threed_enc_fn(
                    state,
                    num_output=self.total_params,
                    reuse=False)
                self.E_var.extend(threeD_var)
            else:
                delta_theta, _ = threed_enc_fn(
                    state, num_output=self.total_params, reuse=True)

            # Compute new theta
            theta_here = theta_prev + delta_theta
            # cam = N x 3, pose N x self.num_theta, shape: N x 10
            cams = theta_here[:, :self.num_cam]
            poses = theta_here[:, self.num_cam:(self.num_cam + self.num_theta)]
            shapes = theta_here[:, (self.num_cam + self.num_theta):]
            # Rs_wglobal is Nx24x3x3 rotation matrices of poses
            verts, joints, pred_Rs = self.smpl(shapes, poses, get_skin=True)

                # --- Compute losses:
            '''
            dict_test = {
                "img": self.image,
                "seg_gt": self.seg_gt,
                "image_feat": self.img_feat
            }

            with tf.Session() as sess:
                res = sess.run(dict_test)
                print("img.shape", res["img"].shape)
                print("seg_gt.shape", res["seg_gt"].shape)
                #print("verts.shape", res["verts"].shape)
                print("image_feat.shape", res["image_feat"].shape)

                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(2)
                axs[0].imshow(res["img"][0])
                seg_np = np.squeeze(res["seg_gt"][0], axis=2)
                axs[1].imshow(seg_np)
                plt.show()
            '''

            pred_kp = batch_orth_proj_idrot(
                joints, cams, name='proj2d_stage%d' % i)
            if(not self.use_mesh_repro_loss):
                loss_kps.append(self.generator_kp_loss_weight * self.keypoint_loss(
                        self.kp_gt, pred_kp))
            if(self.use_mesh_repro_loss):
                silhouette_gt = tf.where(tf.greater(self.seg_gt, 0.))[:, :3]
                silhouette_pred = reproject_vertices(verts, cams,
                                                     self.image.shape.as_list()[1:3], name='mesh_reproject_stage%d' % i)
                self.silhouette_pred = silhouette_pred
                self.silhouette_gt = silhouette_gt

                repro_loss, ba_dist, ab_dist, self.sil_pts_gt, self.sil_pts_pred = self.mesh_repro_loss(
                    silhouette_gt, silhouette_pred, self.batch_size, name='mesh_repro_loss' % i)
                repro_loss_scaled = repro_loss * self.mr_loss_weight

                loss_mr.append(repro_loss_scaled)

            pred_Rs = tf.reshape(pred_Rs, [-1, 24, 9])
            if self.use_3d_label:
                loss_poseshape, loss_joints = self.get_3d_loss(
                    pred_Rs, shapes, joints)
                loss_3d_params.append(loss_poseshape)
                loss_3d_joints.append(loss_joints)

            # Save pred_rotations for Discriminator
            fake_joints.append(joints)
            fake_shapes.append(shapes)

            # Save things for visualiations:
            self.all_verts.append(tf.gather(verts, self.show_these))
            #if(not self.use_mesh_repro_loss):
            self.all_pred_kps.append(tf.gather(pred_kp, self.show_these))
            if(self.use_mesh_repro_loss):
                self.all_pred_silhouettes.append(tf.gather(silhouette_pred, self.show_these))
            self.all_pred_cams.append(tf.gather(cams, self.show_these))

            # Finally update to end iteration.
            theta_prev = theta_here

        if not self.encoder_only:
            self.setup_critic(fake_joints, fake_shapes)

        # Gather losses.
        with tf.name_scope("gather_generator_loss"):
            # Just the last loss.
            self.generator_loss_kp = loss_kps[-1]
            if self.use_mesh_repro_loss:
                self.generator_loss_mr = loss_mr[-1] 
            else:
                self.generator_loss_mr = 0

            if self.encoder_only:
                self.generator_loss = self.generator_loss_kp + self.generator_loss_mr
            else:
                self.generator_loss = (-self.critic_loss_weight * self.critic_loss_fake)\
                                         + self.generator_loss_kp\
                                         + self.generator_loss_mr

            if self.use_3d_label:
                self.generator_loss_3d = loss_3d_params[-1]
                self.generator_loss_3d_joints = loss_3d_joints[-1]

                self.generator_loss += (self.generator_loss_3d + self.generator_loss_3d_joints)

        if not self.encoder_only:
            with tf.name_scope("gather_critic_loss"):
                self.critic_loss = self.critic_loss_weight * (
                    (self.critic_loss_fake - self.critic_loss_real)
                )
                    #* self.c_lamda *  )

        # For visualizations, only save selected few into:
        # B x T x ...
        self.all_verts = tf.stack(self.all_verts, axis=1)
        #if(not self.use_mesh_repro_loss):
        self.all_pred_kps = tf.stack(self.all_pred_kps, axis=1)
        self.show_kps = tf.gather(self.kp_gt, self.show_these)
        if(self.use_mesh_repro_loss):
            #self.all_pred_silhouettes = tf.stack(self.all_pred_silhouettes, axis=1)
            self.show_segs = tf.gather(self.seg_gt, self.show_these)
        self.all_pred_cams = tf.stack(self.all_pred_cams, axis=1)
        self.show_imgs = tf.gather(self.image, self.show_these)


        '''
        if(not self.use_mesh_repro_loss):
            with tf.Session() as sess:
                print("kp_gt: ", self.kp_gt.eval(session=sess))
                print("show_these: ", self.show_these.eval(session=sess))
        '''
        # Don't forget to update batchnorm's moving means.
        print('collecting batch norm moving means!!')
        bn_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if bn_ops:
            self.generator_loss = control_flow_ops.with_dependencies(
                [tf.group(*bn_ops)], self.generator_loss)

        # Setup optimizer
        print('Setting up optimizer..')
        critic_optimizer = self.critic_optimizer(self.critic_lr)
        generator_optimizer = self.generator_optimizer(self.generator_lr)

        #with tf.Session() as sess:
        #    writer = tf.summary.FileWriter("./logs", sess.graph)
        #self.setup_summaries(loss_kps) # remove later
        self.generator_opt = generator_optimizer.minimize(
            self.generator_loss, global_step=self.global_step, var_list=self.E_var)
        if not self.encoder_only:
            self.critic_opt = critic_optimizer.minimize(self.critic_loss, var_list=self.D_var)

        self.setup_summaries(loss_kps)

        print('Done initializing trainer!')

    def setup_summaries(self, loss_kps):
        self.summary_loss = self.generator_loss 

        # Prepare Summary
        always_report = [
            tf.summary.scalar("loss/generator_loss_kp_noscale",
                              self.generator_loss_kp / self.generator_kp_loss_weight),
            tf.summary.scalar("loss/generator_loss_mr_noscale", self.generator_loss_mr /
                              self.generator_kp_loss_weight),
            tf.summary.scalar("loss/generator_loss", self.generator_loss),
            tf.summary.scalar("loss/losses", self.summary_loss)
        ]

        if not self.encoder_only:
            always_report.extend([
                tf.summary.scalar("loss/critic_loss", self.critic_loss),
                tf.summary.scalar("loss/critic_loss_fake", self.critic_loss_fake),
                tf.summary.scalar("loss/critic_loss_real", self.critic_loss_real),
            ])

        # loss at each stage.
        for i in np.arange(self.num_stage):
            name_here = "loss/generator_losses_noscale/kp_loss_stage%d" % i
            always_report.append(
                tf.summary.scalar(name_here, loss_kps[i] / self.generator_kp_loss_weight))

        self.summary_op_always = tf.summary.merge(always_report)

    def setup_critic(self, fake_joints, fake_shapes):
        #TODO check if correct
        all_fake_joints = tf.concat(fake_joints, 0)

        print('real_joints', self.joints)
        print('fake_joints', all_fake_joints)

        combined_joints = tf.concat([tf.squeeze(self.joints), all_fake_joints], 0,
                               name="combined_joints")

        all_fake_shapes = tf.concat(fake_shapes, 0)
        combined_shapes = tf.concat([self.real_shapes, all_fake_shapes], 0, name="combined_shape")

        disc_input = {
            'weight_decay': self.d_wd,
            'shapes': combined_shapes,
            'joints': combined_joints
        }

        self.d_out, self.D_var = Critic_network(**disc_input, C_matrix=self.C,
                                                batch_size=self.batch_size)
        self.d_out_real, self.d_out_fake = tf.split(self.d_out, 2)

        # Compute losses:
        with tf.name_scope("comp_critic_loss"):
            self.critic_loss_fake = losses.compute_weighted_loss(self.d_out_fake)
            self.critic_loss_real = losses.compute_weighted_loss(self.d_out_real)

    def get_3d_loss(self, Rs, shape, joints):
        """
        Rs is N x 24 x 3*3 rotation matrices of pose
        Shape is N x 10
        joints is N x 19 x 3 joints

        Ground truth:
        self.poseshape_loader is a long vector of:
           relative rotation (24*9)
           shape (10)
           3D joints (14*3)
        """
        Rs = tf.reshape(Rs, [self.batch_size, -1])
        params_pred = tf.concat([Rs, shape], 1, name="prep_params_pred")
        # 24*9+10 = 226
        gt_params = self.poseshape_loader[:, :226]
        loss_poseshape = self.e_3d_weight * compute_3d_loss(
            params_pred, gt_params, self.has_gt3d_smpl)
        # 14*3 = 42
        gt_joints = self.poseshape_loader[:, 226:]
        pred_joints = joints[:, :14, :]
        # Align the joints by pelvis.
        pred_joints = align_by_pelvis(pred_joints)
        pred_joints = tf.reshape(pred_joints, [self.batch_size, -1])
        gt_joints = tf.reshape(gt_joints, [self.batch_size, 14, 3])
        gt_joints = align_by_pelvis(gt_joints)
        gt_joints = tf.reshape(gt_joints, [self.batch_size, -1])

        loss_joints = self.e_3d_weight * compute_3d_loss(
            pred_joints, gt_joints, self.has_gt3d_joints)

        return loss_poseshape, loss_joints

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

    def draw_results(self, result, writer):
        import io
        import matplotlib.pyplot as plt
        import cv2

        # This is B x H x W x 3
        imgs = result["input_img"]
        segs_gt = result["seg_gt"]
        # B x 19 x 3
        gt_kps = result["gt_kp"]
        if self.data_format == 'NCHW':
            imgs = np.transpose(imgs, [0, 2, 3, 1])
        # This is B x T x 6890 x 3
        est_verts = result["e_verts"]
        # B x T x 19 x 2
        joints = result["joints"]
        # B x T x 3
        cams = result["cam"]

        img_summaries = []

        if(not self.use_mesh_repro_loss):
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

            vis_sum = tf.Summary.Image(
                encoded_image_string=sio.getvalue(),
                height=combined.shape[0],
                width=combined.shape[1])
            sio.flush()
            sio.close()
            plt.close()
            print("img_summaries.append")
            img_summaries.append(
                tf.Summary.Value(tag="vis_images/%d" % img_id, image=vis_sum))

        img_summary = tf.Summary(value=img_summaries)
        writer.add_summary(
            img_summary, global_step=result['step'])

    def train(self):
        print('started training')
        # For rendering!
        self.renderer = vis_util.SMPLRenderer(
            img_size=self.img_size,
            face_path=self.config.smpl_face_path)

        step = 0
        val_step = 0

        epoch = 0
        time_diff = []
        print('...')
        for epoch in range(self.max_epoch):

            for
                ####################################################
                # Train Pose Prediction Network
                ####################################################

                t0 = time()
                result = sess.run(fetch_dict)
                t1 = time()

                self.summery_writer_train.add_summary(
                    result['summary'], global_step=result['step'])

                generator_loss = result['generator_loss']
                step = result['step']
                time_diff.append(t1-t0)
                epoch = float(step) / self.num_itr_per_epoch
                if self.encoder_only:
                    print("itr %d/(epoch %.1f): time %g, Enc_loss: %.4f" %
                          (step, epoch, t1 - t0, generator_loss))
                else:
                    critic_loss = result['critic_loss']
                    print("itr %d/(epoch %.1f): time %g, Enc_loss: %.4f, Disc_loss: %.4f"
                        % (step, epoch, t1 - t0, generator_loss, critic_loss))

                if step % self.log_img_step == 0:
                    if not self.encoder_only:
                        self.summery_writer_train.add_summary(
                            result['summary_occasional'],
                            global_step=result['step'])
                    print("DRAWING")
                    self.draw_results(result, self.summery_writer_train)

                self.summery_writer_train.flush()

                ####################################################
                # Train Critic Network
                ####################################################
                if not self.encoder_only:
                    fetch_dict = {
                        "critic_opt": self.critic_opt,
                        "critic_loss": self.critic_loss
                        }

                    t0 = time()
                    critic_result = sess.run(fetch_dict)
                    t1 = time()

                    print("Training critic - time: %g, loss: %.4f" % (t1 - t0,
                                                                  critic_result["critic_loss"]))
                    critic_feed_dict = {self.summary_loss: critic_result['critic_loss']}
                    critic_summ = sess.run(self.summary_op_always,
                                           feed_dict=critic_feed_dict)

                    self.summary_writer_critic.add_summary(critic_summ,
                                                           result['step'])

                ####################################################
                # Validation
                ####################################################

                if self.use_val:
                    if step % self.num_itr_per_epoch == 0:
                        val_step += 1
                        print("validation step")

                        sess.run(self.iterator_init_op_val)
                        generator_loss_val = 0
                        for i in range(int(self.num_itr_per_epoch)):

                            fetch_dict = {
                                "generator_loss": self.generator_loss,
                                }

                            t0 = time()
                            result = sess.run(fetch_dict)
                            t1 = time()
                            val_loss = result['generator_loss']
                            generator_loss_val = generator_loss_val + val_loss
#                        if val_step % self.log_val_img_step == 0:
#                            self.draw_results(result, self.summary_writer_val)
                        print("validation generator_loss mean:", generator_loss_val/self.num_itr_per_epoch)

                        sum_result = sess.run(self.summary_op_always,
                                  feed_dict = {self.generator_loss:
                                               generator_loss_val/self.num_itr_per_epoch})
                        self.summary_writer_val.add_summary(sum_result,
                                                                global_step=step)

                        self.summary_writer_val.flush()
                        sess.run(self.iterator_init_op_train)

                if epoch > self.max_epoch:
                    break

                step += 1

        print('Finish training on %s' % self.model_dir)
        print('Average time per iteration was', np.mean(time_diff)) 
