from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config as config
from src.config import get_config, prepare_dirs, save_config
from src.data_loader import DataLoader
#from src.RunModel import RunModel
#from src.util.load_data import example_run
from src.models import Encoder_resnet, Critic_network
import deepdish as dd
from opendr.renderer import ColoredRenderer 
from opendr.lighting import LambertianPointLight 
from opendr.camera import ProjectPoints 
from src.tf_smpl.batch_smpl import SMPL

from os.path import join, dirname
#import src.util.data_utils as du

def plot_verts(verts, img_size, smpl_face_path, a,b,c, axarr, k, l):
    print(verts.shape)
    ## Create OpenDR renderer
    renderer = vis_util.SMPLRenderer(
        img_size=img_size,
        face_path=smpl_face_path)
    w = 224
    h = 224
    print(a,b,c)
    camera = ProjectPoints(rt=[a,b,c],
                           t=np.array([0, 0, 1.2]),
                           f=np.array([w,w])/2.,
                           c=np.array([w,h])/2.,
                           k=np.zeros(5))
    #use_cam = ProjectPoints(
    #    f=cam[0] * np.ones(2),
    #    rt=np.zeros(3),
    #    t=np.zeros(3),
    #    k=np.zeros(5),
    #    c=cam[1:3])

    rend_img = renderer(verts, camera)
    print("k",k,"l",l)
    axarr[k,l].imshow(rend_img)
    axarr[k,l].axis('off')
    return axarr

def main(config):
    prepare_dirs(config)

    # Load data on CPU
    with tf.device("/cpu:0"):
        data_loader = DataLoader(config)
        dataset = data_loader.load()
        smpl_dataset = data_loader.get_smpl_loader()

    m = SMPL(config.smpl_model_path)


    dataset = smpl_dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(8)


#    smpl_dataset = smpl_dataset.shuffle(buffer_size=1000).repeat()
#    smpl_dataset = smpl_dataset.batch(1)

#    big_dataset = tf.data.Dataset.zip((dataset, smpl_dataset))

#    result = critic([a,b,c], training=True)
#    print(result)
#    iterator = dataset.make_one_shot_iterator()

#    trainer = HMRTrainer(config, dataset, smpl_loader)
#    save_config(config)
#    trainer.train()
#    smpl_dataset = smpl_dataset.shuffle(buffer_size=1000).repeat()
#    smpl_dataset = smpl_dataset.batch(1)

#    big_dataset = tf.data.Dataset.zip((dataset, smpl_dataset))

#    result = critic([a,b,c], training=True)
#    print(result)

#    init_mean = load_mean_param(config)
    faces = np.load("src/tf_smpl/smpl_faces.npy")
    with tf.GradientTape() as gen_tape:
        for shape, joints, beta, verts in dataset:
            #image, seg_gt, kps = a
            #features = model(image)
            #print(features.shape)
            #state = tf.concat([features, init_mean], 1)
            #print(state)
            #break
            #pose, shape = b
            #print(image.shape)
            #print(pose.shape)
            #print(shape.shape)
            #print(kps.shape)
            f, axarr = plt.subplots(8,4, figsize=(4, 8), dpi=200)
            for j in range(8):
                plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)
                k = -1
                a = -1.5
                b = 2.5
                c = -2
                axarr = plot_verts(verts[j], config.img_size, config.smpl_face_path, a,b,c, axarr, j, 0)
                a = 0.5
                b = 1.5
                c = -0.5
                axarr = plot_verts(verts[j], config.img_size, config.smpl_face_path, a,b,c, axarr, j, 1)
                a = 0.5
                b = 1.5
                c = -2
                axarr = plot_verts(verts[j], config.img_size, config.smpl_face_path, a,b,c, axarr, j, 2)
                a = 0.5
                b = -1.5
                c = -2
                axarr = plot_verts(verts[j], config.img_size, config.smpl_face_path, a,b,c, axarr, j, 3)

                #for a in [-2, -1.5, -1, -0.7, -0.5, -0.3, 0., 0.5]:
                #    l = -1
                #    k += 1
                #    for b in [1.5, 2., 2.5]:
                #        for c in [-0.5, -1., -1.5, -2]:
                #            l += 1
                #            print(verts.shape)
                #            ## Create OpenDR renderer
                #            renderer = vis_util.SMPLRenderer(
                #                img_size=config.img_size,
                #                face_path=config.smpl_face_path)
                #            w = 224
                #            h = 224
                #            print(a,b,c)
                #            camera = ProjectPoints(rt=[a,b,c],
                #                                   t=np.array([0, 0, 1.2]),
                #                                   f=np.array([w,w])/2.,
                #                                   c=np.array([w,h])/2.,
                #                                   k=np.zeros(5))
                #            #use_cam = ProjectPoints(
                #            #    f=cam[0] * np.ones(2),
                #            #    rt=np.zeros(3),
                #            #    t=np.zeros(3),
                #            #    k=np.zeros(5),
                #            #    c=cam[1:3])

                #            rend_img = renderer(verts[j], camera)
                #            print("k",k,"l",l)
                #            axarr[k,l].imshow(rend_img)
                #            axarr[k,l].axis('off')
            plt.show()



if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    main(config)
