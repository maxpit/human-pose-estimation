"""
Demo of HMR.

Note that HMR requires the bounding box of the person in the image. The best performance is obtained when max length of the person in the image is roughly 150px.

When only the image path is supplied, it assumes that the image is centered on a person whose length is roughly 150px.
Alternatively, you can supply output of the openpose to figure out the bbox and the right scale factor.

Sample usage:

# On images on a tightly cropped image around the person
python -m demo --img_path data/im1963.jpg
python -m demo --img_path data/coco1.png

# On images, with openpose output
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
from src.tf_smpl.projection import batch_orth_proj_idrot
import src.config
from src.RunModel import RunModel

flags.DEFINE_string('img_path', 'data/im1963.jpg', 'Image to run')
flags.DEFINE_string(
    'json_path', None,
    'If specified, uses the openpose output to crop the image.')

def bidirectional_dist(A, B):
    # get nearest neighbors B of A
    #if (A = tenso)
    nbrs_B = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(A.eval())
    distances_BA, ind_BA = nbrs_B.kneighbors(B.eval())

    # get nearest neighbors A of B
    nbrs_A = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(B.eval())
    distances_AB, ind_AB = nbrs_A.kneighbors(A.eval())

    # compute distances
    dist_BA = tf.norm(tf.to_float(B) - tf.to_float(tf.gather(A, ind_BA[:, 0])), axis=1)
    dist_AB = tf.norm(tf.to_float(A) - tf.to_float(tf.gather(B, ind_AB[:, 0])), axis=1)
    summed_dist_BA = tf.reduce_sum(dist_BA)
    summed_dist_AB = tf.reduce_sum(dist_AB)
    # print(distances_BA.sum()+distances_AB.sum())
    # print(tf.math.add(summed_dist_BA, summed_dist_AB).eval())

    return tf.math.add(summed_dist_BA, summed_dist_AB)

def mesh_reprojection_loss(silhouette_gt, silhouette_pred, name=None):
    """
    ADL4CV
    Computes bidirectional distance between ground truth silhouette and predicted silhouette
    Inputs:
        silhouette_gt:      N x K x 2
        silhouette_pred:    N x K x 2
    """
    with tf.name_scope(name, "mesh_reprojection_loss", [silhouette_gt, silhouette_pred]):
        N = silhouette_gt.shape[0]
        #K = silhouette_gt.shape[1]
      #  silhouette_gt = tf.reshape(silhouette_gt, (-1, 2))
      #  silhouette_pred = tf.reshape(silhouette_pred, (-1, 2))
      #  sil_gt_np = silhouette_gt.eval()
      #  sil_pred_np = silhouette_pred.eval()
        loss = tf.constant(0.)
        for i in range(N):
            loss = tf.math.add(loss, bidirectional_dist(silhouette_gt[i,:,:], silhouette_pred[i,:,:]))
        return loss

def reproject_vertices(proc_param, verts, cam, im_size):
    """
    already works for COCO data
    """
    # im_size = img.shape[:2]
    print("REPROJECT VERTICES: ")
    print("proc_param: ", proc_param)
    print("im_size: ", im_size)
    print("cam before expand: ", cam)
    cam_ = tf.expand_dims(cam, 0)
    verts_ = tf.expand_dims(verts, 0)
    print("cam after expand: ", cam_)
    verts_projected = batch_orth_proj_idrot(tf.cast(verts_, tf.float32), tf.cast(cam_, tf.float32))
    verts_projected = ((verts_projected + 1) * 0.5 * proc_param['scale']) * im_size  # proc_param['img_size'] instead of im_size?

    img_size = proc_param['img_size']

    print("img_size: ", img_size)
    margin = int(img_size / 2)
    undo_scale = 1. / np.array(proc_param['scale'])
    verts_original = (verts_projected + proc_param['start_pt'] - margin) * undo_scale

    return verts_original # shape num_verts x 2

def reproject_vertices_prototyping(proc_param, verts, cam, im_size):
    """
    TODO: adapt to all training data
    """
    print("REPROJECT VERTICES: ")
    print("proc_param: ", proc_param)
    print("im_size: ", im_size)
    print("cam before expand: ", cam)
    scale = proc_param['scale']
    img_size = proc_param['img_size']
    # im_size = img.shape[:2]
    #cam_ = tf.expand_dims(cam, 0)
    #print("cam after expand: ", cam_)
    #verts_ = tf.expand_dims(verts, 0)
    verts_projected = batch_orth_proj_idrot(tf.cast(verts, tf.float32), tf.cast(cam, tf.float32))
    print("shape verts_projected: ", verts_projected)
    #scale_prototyping = np.array([1.9649, scale])
    verts_projected = ((verts_projected+1)* 0.5 ) * img_size  # proc_param['img_size'] instead of im_size?


    margin = int(img_size / 2)

    margin = (np.array(im_size)/2).astype(int)
    undo_scale = 1. / np.array(proc_param['scale'])

    #verts_original = (verts_projected + proc_param['start_pt'] - margin) / scale_prototyping

  #  return verts_original # shape num_verts x 2
    return verts_projected

def visualize_sil_reprojection(img, orig_img, proc_param, verts, cam, sess):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    #im_silhouette = mpimg.imread('data/cropped_3_gt.jpg')
    #print(im_silhouette.shape)
    #im_silhouette = np.floor(im_silhouette)
    img_vis_0 = ((img[0] * 0.5) + 0.5) * 255
    img_vis_1 = ((img[1] * 0.5) + 0.5) * 255
    img_vis_2 = ((img[2] * 0.5) + 0.5) * 255
    verts_orig = reproject_vertices_prototyping(proc_param, verts, cam, img.shape[1:2])
    # print(verts_orig)
    # verts_pixel = tf.cast(verts_orig, tf.int32)
    verts_pixel = verts_orig
    #print(img_vis)
    #print(img_vis.shape)
    plt.figure(1)
    plt.subplot(211)
    plt.imshow(img_vis_0.astype(int))
    #print(verts_pixel)
    #verts_pixel_np = np.unique(verts_pixel.eval(session=sess), axis=1)
    #print(verts_pixel_np.shape)
    plt.plot(verts_pixel[0, :, 0].eval(session=sess), verts_pixel[0, :, 1].eval(session=sess), 'bs', markersize=0.1)
    #plt.plot(verts_pixel_np[0, :, 0], verts_pixel_np[0, :, 1], 'bs')

    plt.subplot(212)
    plt.imshow(img_vis_1.astype(int))
    plt.plot(verts_pixel[1, :, 0].eval(session=sess), verts_pixel[1, :, 1].eval(session=sess), 'bs', markersize=0.1)

    #plt.subplot(213)
    #plt.imshow(img_vis_2.astype(int))
    #plt.plot(verts_pixel[2, :, 0].eval(session=sess), verts_pixel[2, :, 1].eval(session=sess), 'bs', markersize=0.1)


    plt.show()


def visualize(img, proc_param, joints, verts, cam):
    """
    Renders the result in original image coordinate frame.
    """
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])

    #print("cam_for_render: ", cam_for_render)
    #print("vert_shifted: ", vert_shifted)
    #print("joints_orig: ", joints_orig)
    #print("image shape: ", img.shape)

    # Render results
    skel_img = vis_util.draw_skeleton(img, joints_orig)
    rend_img_overlay = renderer(
        vert_shifted, cam=cam_for_render, img=img, do_alpha=True)
    rend_img = renderer(
        vert_shifted, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp1 = renderer.rotated(
        vert_shifted, 60, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp2 = renderer.rotated(
        vert_shifted, -60, cam=cam_for_render, img_size=img.shape[:2])

    import matplotlib.pyplot as plt
    # plt.ion()
    plt.figure(1)
    plt.clf()
    plt.subplot(231)
    plt.imshow(img)
    plt.title('input')
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(skel_img)
    plt.title('joint projection')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(rend_img_overlay)
    plt.title('3D Mesh overlay')
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(rend_img)
    plt.title('3D mesh')
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(rend_img_vp1)
    plt.title('diff vp')
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(rend_img_vp2)
    plt.title('diff vp')
    plt.axis('off')
    plt.draw()
    plt.show()
    # import ipdb
    # ipdb.set_trace()


def preprocess_image(img_path, json_path=None):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    print("PREPROCESSING")
    print("img shape: ", img.shape)

    if json_path is None:
        if np.max(img.shape[:2]) != config.img_size:
            print('Resizing so the max image size is %d..' % config.img_size)
            scale = (float(config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
    else:
        scale, center = op_util.get_bbox(json_path)

    print("scale: ", scale)
    print("center: ", center)

    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)
    print("crop 1: ", crop)
    print("crop 1 shape: ", crop.shape)
    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    print("crop 2: ", crop)
    print("crop 2 shape: ", crop.shape)
    return crop, proc_param, img


def main(img_path, json_path=None):
    sess = tf.Session()
    model = RunModel(config, sess=sess)
    import matplotlib.image as mpimg


    for i in range(1, 4):
    #original_img = mpimg.imread(img_path)
        path = 'data/coco{}.png'.format(i)
        original_img = mpimg.imread(path)
        input_img, proc_param, img = preprocess_image(path, json_path)
        input_img_for_vis = input_img
        # Add batch dimension: 1 x D x D x 3
        if(i==1):
            input_imgs = np.expand_dims(input_img, 0)
        else:
            input_imgs = np.append(input_imgs, np.expand_dims(input_img, 0), axis=0)
    print("input_imgs.shape: ", input_imgs.shape)

    for i in range(3):
    # Theta is the 85D vector holding [camera, pose, shape]
    # where camera is 3D [s, tx, ty]
    # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
    # shape is 10D shape coefficients of SMPL
        joints, verts, cams, joints3d, theta = model.predict(
            np.expand_dims(input_imgs[i], 0), get_theta=True)
        if(i == 0):
            #verts_all = tf.expand_dims(verts, 0)
            #cams_all = tf.expand_dims(cams, 0)
            verts_all = verts
            cams_all = cams
        else:
            verts_all = tf.concat([verts_all, verts], axis=0)
            cams_all = tf.concat([cams_all, cams], axis=0)
    #print("input img: ", input_img.shape)
    #np.set_printoptions(threshold=sys.maxsize)

    #visualize(img, proc_param, joints[0], verts[0], cams[0])
    visualize_sil_reprojection(input_imgs, original_img, proc_param, verts_all, cams_all, sess)

if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL
    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    main(config.img_path, config.json_path)
