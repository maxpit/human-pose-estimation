from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np

import cv2
import tensorflow as tf

from src.predictor import Predictor

from src.util import renderer as vis_util
from src.util import image as img_util


def preprocess_image(img, config):
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if np.max(img.shape[:2]) != config.img_size:
        scale = (float(config.img_size) / np.max(img.shape[:2]))
    else:
        scale = 1.
    center = np.round(np.array(img.shape[:2]) / 2).astype(int)
    # image center in (x,y)
    center = center[::-1]

    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)
    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img

def main(config):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

    renderer = vis_util.SMPLRenderer(
        img_size=config.img_size,
        face_path=config.smpl_face_path)

    config.checkpoint_dir = "training_checkpoints_125_epochs_lspe"
    predictor = Predictor(config)
    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("preview", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    draw_skel = False
    draw_mesh = True
    rotate_img = False
    show_both = False

    while rval:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        input_img, proc_param, img = preprocess_image(frame, config)
        verts, cam, joints = predictor.do_prediction(input_img)
        cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
            proc_param, np.squeeze(verts), np.squeeze(cam), np.squeeze(joints)[:,:2], img_size=frame.shape[:2])

        if tf.math.is_nan(joints_orig).numpy().any():
            print("nothing found")
            rend_img = frame
        else:
            if draw_skel:
                rend_img = vis_util.draw_skeleton(frame, joints_orig)
            if draw_mesh:
                if rotate_img:
                    rend_img = renderer.rotated(vert_shifted, 60, cam=cam_for_render, img_size=frame.shape[:2])
                else:
                    rend_img = renderer(vert_shifted, cam_for_render, frame, True)
                    if show_both:
                        img2 = renderer.rotated(vert_shifted, 60, cam=cam_for_render, img_size=frame.shape[:2])
                        rend_img = np.concatenate((rend_img, img2), axis=1)


        cv2.imshow("preview", rend_img)
        for i in range(5):
            rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
        if key == ord('s'):
            draw_skel = True
            draw_mesh = False
            rotate_img = False
            show_both = False

        if key == ord('m'):
            draw_skel = False
            draw_mesh = True
            rotate_img = False
            show_both = False

        if key == ord('r'):
            draw_skel = False
            draw_mesh = True
            rotate_img = True
            show_both = False

        if key == ord('b'):
            draw_skel = False
            draw_mesh = True
            rotate_img = False
            show_both = True

    cv2.destroyWindow("preview")

if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    main(config)
