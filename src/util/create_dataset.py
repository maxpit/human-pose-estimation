import numpy as np
import tensorflow as tf
import scipy.io as sio
import re
import matplotlib.pyplot as plt

from glob import glob
from os.path import basename
from .common import ImageCoder
from os.path import join, dirname

file_dir = '/home/valentin/Code/ADL/human-pose-estimation/data/'

lsp_dir = file_dir + 'lsp/'
lsp_e_dir = file_dir + 'lspet_dataset/'
lsp_im = lsp_dir + 'images/'
lsp_e_im = lsp_e_dir + 'images/'
lsp_seg = file_dir + 'upi-s1h/data/lsp/'
lsp_e_seg = file_dir + 'upi-s1h/data/lsp_extended/'
mpii_dir = join(file_dir, 'upi-s1h/data/mpii/')
mpii_poses_dir = join(mpii_dir, 'poses.npz')

def load_mat(fname):
    import scipy.io as sio
    res = sio.loadmat(fname)
    # this is 3 x 14 x 2000
    return res['joints']

def _add_to_tfrecord(img_path, gt_path, label, writer, is_lsp_ext=False, is_mpii=False):

    if is_lsp_ext:
        visible = label[2, :].astype(bool)
    else:
        visible = np.logical_not(label[2, :])
        label[2, :] = visible.astype(label.dtype)

    min_pt = np.min(label[:2, visible], axis=1)
    max_pt = np.max(label[:2, visible], axis=1)
    center = (min_pt + max_pt) / 2.

    with tf.io.gfile.GFile(img_path, 'rb') as f:
        image_data = f.read()

    with tf.io.gfile.GFile(gt_path, 'rb') as f:
        seg_data = f.read()

    img = tf.image.decode_jpeg(image_data)
    # lsp ext segentation data has 3 channels so reducing it to one to match lsp
    if is_lsp_ext or is_mpii:
        seg_gt = tf.image.decode_jpeg(seg_data)
        seg_gt = tf.expand_dims(seg_gt[:,:,0], 2)
        seg_data = tf.image.encode_jpeg(seg_gt).numpy()
    #print("image", len(image_data))
    #print("seg", len(seg_data))
    #print("label", label)
    #print("center", center)
    #print("width,height", img.shape)
    #print("width,height", seg_gt.shape)
    #img = np.array(io.imread(img_path))
    #+gt = np.array(io.imread(gt_path))
    #gt = coder.decode_jpeg(seg_data)

    #f, axarr = plt.subplots(1,2)
    #seg_gt = tf.concat([seg_gt, seg_gt, seg_gt], axis=2)
    #plt.imshow(seg_gt)
    #plt.scatter(x=label[0,:], y=label[1,:], c=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
    #                                           'C8', 'C9', 'b', 'g', 'r', 'y'], s=4)
    #axarr[1].imshow(seg_gt)
    #plt.show()
    #img_raw = img.tostring()
    #gt_raw = gt.tostring()
    add_face = False
    if label.shape[1] == 19:
        add_face = True
        # Split and save facepts on it's own.
        face_pts = label[:, 14:]
        label = label[:, :14]

    feat_dict = {
        'image/height': _int64_feature(img.shape[0]),
        'image/width': _int64_feature(img.shape[1]),
        'image/center': _int64_feature(center.astype(np.int)),
        'image/x': _float_feature(label[0, :].astype(np.float)),
        'image/y': _float_feature(label[1, :].astype(np.float)),
        'image/visibility': _int64_feature(label[2, :].astype(np.int)),
        'image/filename': _bytes_feature(tf.compat.as_bytes(basename(img_path))),
        'image/encoded': _bytes_feature(tf.compat.as_bytes(image_data)),
        'image/seg_gt': _bytes_feature(tf.compat.as_bytes(seg_data))
    }
    if add_face:
        # 3 x 5
        feat_dict.update({
            'image/face_pts':
            _float_feature(face_pts.ravel().astype(np.float))
        })

    example = tf.train.Example(features=tf.train.Features(feature=feat_dict))

    writer.write(example.SerializeToString())

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def create (tfrecords_filename, filename_pairs, dataset="lsp"):
    writer = tf.io.TFRecordWriter(tfrecords_filename)
    is_lsp_ext = False
    is_mpii = False

    if dataset == "lsp":
        mat_dir = lsp_dir
        # Load labels 3 x 14 x N
        labels = load_mat((mat_dir+'joints.mat'))
    elif dataset == "lsp_ext":
        mat_dir = lsp_e_dir
        is_lsp_ext = True
        # Load labels 3 x 14 x N
        labels = load_mat((mat_dir+'joints.mat'))
    else:
        is_mpii = True
        labels = np.load(mpii_poses_dir)['poses']
        # Mapping from MPII joints to LSP joints (0:13). In this roder:
        _COMMON_JOINT_IDS = [
            0,  # R ankle
            1,  # R knee
            2,  # R hip
            3,  # L hip
            4,  # L knee
            5,  # L ankle
            10,  # R Wrist
            11,  # R Elbow
            12,  # R shoulder
            13,  # L shoulder
            14,  # L Elbow
            15,  # L Wrist
            8,  # Neck top
            9,  # Head top
        ]
        labels = labels[:,_COMMON_JOINT_IDS,:]

    if labels.shape[0] != 3:
        labels = np.transpose(labels, (1, 0, 2))

    print(labels.shape)
    for i in range(len(filename_pairs)):
        current_file = int(re.findall('\d+',filename_pairs[i][0])[0])
        if is_mpii:
            current_file = i+1
        _add_to_tfrecord(
                filename_pairs[i][0],
                filename_pairs[i][1],
                labels[:, :, current_file-1],
                writer,
                is_lsp_ext,
                is_mpii)

    writer.close()

def get_filename_pairs_single():
    im1 = lsp_im + 'im0007.jpg'
    seg1 = lsp_seg + 'im0007_segmentation.png'

    filename_pair = [(im1, seg1)]

    return filename_pair

def get_filename_pairs_few(few_count = 9):
    return get_filename_pairs_lsp()[:few_count]

def get_filename_pairs_lsp():
    all_images = sorted([f for f in glob((lsp_im+'*.jpg'))])
    all_seg_gt = sorted([f for f in glob((lsp_seg+'im[0-9][0-9][0-9][0-9]_segmentation.png'))])
    filename_pairs = tuple(np.vstack((all_images, all_seg_gt)).transpose())

    return filename_pairs

def get_filename_pairs_lspe():
    all_images = sorted([f for f in glob((lsp_e_im+'*.jpg'))])
    all_seg_gt = sorted([f for f in glob((lsp_e_seg+'im[0-9][0-9][0-9][0-9][0-9]_segmentation.png'))])

    ss = []
    for s in all_seg_gt:
        current_file = int(re.findall('\d+',s)[1])
        ss.append(current_file)

    all_images = [all_images[index-1] for index in ss]
    filename_pairs = tuple(np.vstack((all_images, all_seg_gt)).transpose())

    return filename_pairs


def get_filename_pairs_mpii():
    all_images = sorted([f for f in glob((mpii_dir+'images/[0-9][0-9][0-9][0-9][0-9].png'))])
    all_seg_gt = sorted([f for f in glob((mpii_dir+'images/[0-9][0-9][0-9][0-9][0-9]_segmentation.png'))])
    filename_pairs = tuple(np.vstack((all_images, all_seg_gt)).transpose())

    return filename_pairs

