import numpy as np
import skimage.io as io
import tensorflow as tf

file_dir = '/home/valentin/Code/ADL/human-pose-estimation/data'

lsp_im = file_dir + '/lsp_dataset/images/'
lsp_seg = file_dir + '/upi-s1h/data/lsp/'

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create (tfrecords_filename, filename_pairs):
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    for img_path, gt_path in filename_pairs:

        img = np.array(io.imread(img_path))
        gt = np.array(io.imread(gt_path))

        img_raw = img.tostring()
        gt_raw = gt.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(img.shape[0]),
            'width': _int64_feature(img.shape[1]),
            'image_raw': _bytes_feature(img_raw),
            'mask_raw': _bytes_feature(gt_raw)}))

        writer.write(example.SerializeToString())

    writer.close()


def get_filename_pairs_single():
    im1 = lsp_im + 'im0001.jpg'
    seg1 = lsp_seg + 'im0001_segmentation.png'

    filename_pairs = [(im1, seg1)]

    return filename_pairs

def get_filename_pairs_few(few_count = 9):
    filename_pairs = []

    for i in range(1, few_count):
        im = lsp_im+'im000'+str(i)+'.jpg'
        gt = lsp_seg+'im000'+str(i)+'_segmentation.png'
        filename_pairs.append((im,gt))

    return filename_pairs

def get_filename_pairs_lsp():
    
    return filename_pairs

def get_filename_pairs_lspe():

    return filename_pairs


def get_filename_pairs_mpii():

    return filename_pairs

def get_filename_pairs_all():

    return filename_pairs
