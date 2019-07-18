import numpy as np
import tensorflow as tf
import scipy.io as sio
import re

from glob import glob
from os.path import basename
from .common import ImageCoder

file_dir = '/home/valentin/Code/ADL/human-pose-estimation/data/'

lsp_dir = file_dir + 'lsp_dataset/'
lsp_e_dir = file_dir + 'lspet_dataset/'
lsp_im = lsp_dir + 'images/'
lsp_e_im = lsp_e_dir + 'images/'
lsp_seg = file_dir + 'upi-s1h/data/lsp/'
lsp_e_seg = file_dir + 'upi-s1h/data/lsp_extended/'
mpi_dir = file_dir + 'mpi_3dhp'

def load_mat(fname):
    import scipy.io as sio
    res = sio.loadmat(fname)
    # this is 3 x 14 x 2000
    return res['joints']

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _add_to_tfrecord(img_path, gt_path, label, writer, is_lsp_ext=False):

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

def create_lsp_set(tfrecords_filename, filename_pair):
    if len(filename_pair) == 10000:
        # LSP-extended is all train.
        train_out = join(out_dir, 'train_%03d.tfrecord')
        package(all_images, labels, train_out, num_shards_train)
    else:
        train_out = join(out_dir, 'train_%03d.tfrecord')

        package(all_images[:1000], labels[:, :, :1000], train_out,
                num_shards_train)

        test_out = join(out_dir, 'test_%03d.tfrecord')
        package(all_images[1000:], labels[:, :, 1000:], test_out,
                num_shards_test)

def create (tfrecords_filename, filename_pairs, is_lsp_ext=False):
    writer = tf.io.TFRecordWriter(tfrecords_filename)

    # Load labels 3 x 14 x N
    labels = load_mat((lsp_e_dir+'joints.mat'))
    if labels.shape[0] != 3:
        labels = np.transpose(labels, (1, 0, 2))

    for i in range(len(filename_pairs)):
        print(filename_pairs[i][0])
        current_file = int(re.findall('\d+',filename_pairs[i][0])[0])
        _add_to_tfrecord(
                filename_pairs[i][0],
                filename_pairs[i][1],
                labels[:, :, current_file-1],
                writer,
                is_lsp_ext)

    writer.close()

def get_available_mpi_data():
    all_dirs

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

    print(len(all_images))
    print(len(all_seg_gt))
    ss = []
    for s in all_seg_gt:
        print(s)
        current_file = int(re.findall('\d+',s)[1])
        ss.append(current_file)

    all_images = [all_images[index] for index in ss]
    filename_pairs = tuple(np.vstack((all_images, all_seg_gt)).transpose()) 

    return filename_pairs


def get_filename_pairs_mpii():

    return filename_pairs

def get_filename_pairs_all():

    return filename_pairs
