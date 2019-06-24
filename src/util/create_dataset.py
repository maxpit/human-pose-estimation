import numpy as np
import skimage.io as io
import tensorflow as tf
import scipy.io as sio

from glob import glob
from os.path import basename
from .common import ImageCoder

file_dir = '/home/valentin/Code/ADL/human-pose-estimation/data/'

lsp_dir = file_dir + 'lsp_dataset/'
lsp_im = lsp_dir + 'images/'
lsp_seg = file_dir + 'upi-s1h/data/lsp/'
mpi_dir = file_dir + 'mpi_3dhp'

def load_mat(fname):
    import scipy.io as sio
    res = sio.loadmat(fname)
    # this is 3 x 14 x 2000
    return res['joints']

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))




def _add_to_tfrecord(img_path, gt_path, label, coder, writer, is_lsp_ext=False):

    if is_lsp_ext:
        visible = label[2, :].astype(bool)
    else:
        visible = np.logical_not(label[2, :])
        label[2, :] = visible.astype(label.dtype)

    min_pt = np.min(label[:2, visible], axis=1)
    max_pt = np.max(label[:2, visible], axis=1)
    center = (min_pt + max_pt) / 2.

    with tf.gfile.FastGFile(img_path, 'rb') as f:
        image_data = f.read()

    with tf.gfile.FastGFile(gt_path, 'rb') as f:
        seg_data = f.read()

    img = coder.decode_jpeg(image_data)
    #img = np.array(io.imread(img_path))
    #+gt = np.array(io.imread(gt_path))
    #gt = coder.decode_jpeg(seg_data)

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


def create (tfrecords_filename, filename_pairs):
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    coder = ImageCoder()
    # Load labels 3 x 14 x N
    labels = load_mat((lsp_dir+'joints.mat'))
    if labels.shape[0] != 3:
        labels = np.transpose(labels, (1, 0, 2))

    for i in range(len(filename_pairs)):
        _add_to_tfrecord(
                filename_pairs[i][0],
                filename_pairs[i][1],
                labels[:, :, i],
                coder,
                writer)
 
    writer.close()

def get_available_mpi_data():
    all_dirs

def get_filename_pairs_single():
    im1 = lsp_im + 'im0001.jpg'
    seg1 = lsp_seg + 'im0001_segmentation.png'

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

    return filename_pairs


def get_filename_pairs_mpii():

    return filename_pairs

def get_filename_pairs_all():

    return filename_pairs
