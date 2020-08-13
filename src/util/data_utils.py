"""
Utils for data loading for training.
"""

from os.path import join
from glob import glob

import tensorflow as tf


def parse_example_proto(example_serialized, has_3d=False):
    """Parses an Example proto.
    It's contents are:

        'image/height'       : _int64_feature(height),
        'image/width'        : _int64_feature(width),
        'image/x'            : _float_feature(label[0,:].astype(np.float)),
        'image/y'            : _float_feature(label[1,:].astype(np.float)),
        'image/visibility'   : _int64_feature(label[2,:].astype(np.int)),
        'image/format'       : _bytes_feature
        'image/filename'     : _bytes_feature
        'image/encoded'      : _bytes_feature
    """
    feature_map = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/seg_gt':  tf.io.FixedLenFeature([], tf.string),
        'image/height':  tf.io.FixedLenFeature([], tf.int64),
        'image/width':   tf.io.FixedLenFeature([], tf.int64),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/center': tf.io.FixedLenFeature((2, 1), dtype=tf.int64),
        'image/visibility': tf.io.FixedLenFeature((1, 14), dtype=tf.int64),
        'image/x': tf.io.FixedLenFeature((1, 14), dtype=tf.float32),
        'image/y': tf.io.FixedLenFeature((1, 14), dtype=tf.float32),
        'image/face_pts': tf.io.FixedLenFeature(
            (1, 15),
            dtype=tf.float32,
            default_value=[
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
            ]),
    }

    features = tf.io.parse_single_example(example_serialized, feature_map)

    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)
    center = tf.cast(features['image/center'], tf.int32)
    fname = tf.cast(features['image/filename'], tf.string)

    face_pts = tf.reshape(tf.cast(features['image/face_pts'], dtype=tf.float32), [3, 5])

    vis = tf.cast(features['image/visibility'], dtype=tf.float32)
    x = tf.cast(features['image/x'], dtype=tf.float32)
    y = tf.cast(features['image/y'], dtype=tf.float32)

    label = tf.concat([x, y, vis], 0)
    label = tf.concat([label, face_pts], 1)

    image_shape = ([height, width, 3])
    annotation_shape = ([height, width, 1])

    image = decode_jpeg(features['image/encoded'],3)
    seg_gt= decode_jpeg(features['image/seg_gt'],1)

    image = tf.reshape(image, image_shape)
    seg_gt= tf.reshape(seg_gt, annotation_shape)

    image_size = [height, width]

    return image, seg_gt, image_size, label, center, fname


def rescale_image(image):
    """
    Rescales image from [0, 1] to [-1, 1]
    Resnet v2 style preprocessing.
    """
    # convert to [0, 1].
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def get_all_files(dataset_dir, datasets, split='train'):
    # Dataset with different name path
    diff_name = ['h36m', 'mpi_inf_3dhp']

    data_dirs = [
        (dataset_dir+'/'+dataset+'.tfrecords')
        for dataset in datasets if dataset not in diff_name
    ]

    print(data_dirs)

    if 'h36m' in datasets:
        data_dirs.append(
            join(dataset_dir, 'tf_records_human36m_wjoints', split,
                 '*.tfrecords'))
    if 'mpi_inf_3dhp' in datasets:
        data_dirs.append(
            join(dataset_dir, 'mpi_inf_3dhp', split, '*.tfrecords'))

    all_files = []
    for data_dir in data_dirs:
        all_files += sorted(glob(data_dir))

    return data_dirs#all_files


def parse_mocap_example(example_serialized):
    """
    Parses a smpl Example proto.
    It's contents are:
        'pose'  : 72-D float
        'shape' : 10-D float
    """
    with tf.name_scope('read_smpl_data'):

        feature_map = {
            'pose': tf.io.FixedLenFeature((72, ), dtype=tf.float32),
            'shape': tf.io.FixedLenFeature((10, ), dtype=tf.float32)
        }

        features = tf.io.parse_single_example(example_serialized, feature_map)
        pose = tf.cast(features['pose'], dtype=tf.float32)
        shape = tf.cast(features['shape'], dtype=tf.float32)

        return pose, shape

def decode_jpeg(image_buffer, ch,  name=None):
    """Decode a JPEG string into one 3-D float image Tensor.
      Args:
        image_buffer: scalar string Tensor.
        name: Optional name for name_scope.
      Returns:
        3-D float Tensor with values ranging from [0, 1).
    """
    with tf.name_scope('decode_jpeg'):
        image = tf.image.decode_jpeg(image_buffer, channels=ch)
        # convert to [0, 1].
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image


def jitter_center(center, trans_max):
    with tf.name_scope('jitter_center'):
        rand_trans = tf.random.uniform( [2, 1], minval=-trans_max, maxval=trans_max, dtype=tf.int32)
        return center + rand_trans


def jitter_scale(image, seg_gt, image_size, keypoints, center, scale_range):

    with tf.name_scope('jitter_scale'):
        scale_factor = tf.random.uniform(
            [1],
            minval=scale_range[0],
            maxval=scale_range[1],
            dtype=tf.float32)
        new_size = tf.cast(tf.cast(image_size, tf.float32) * scale_factor,
                           tf.int32)
        new_image = tf.image.resize(image, new_size)
        new_seg_gt = tf.image.resize(seg_gt, new_size)

        # This is [height, width] -> [y, x] -> [col, row]
        actual_factor = tf.cast(tf.shape(new_image)[:2], tf.float32) / tf.cast(image_size, tf.float32)
        x = keypoints[0, :] * actual_factor[1]
        y = keypoints[1, :] * actual_factor[0]

        cx = tf.cast(center[0], actual_factor.dtype) * actual_factor[1]
        cy = tf.cast(center[1], actual_factor.dtype) * actual_factor[0]

        return new_image, new_seg_gt, tf.stack([x, y]), tf.cast(
            tf.stack([cx, cy]), tf.int32)


def pad_image_edge(image, margin):
    """ Pads image in each dimension by margin, in numpy:
    image_pad = np.pad(image,
                       ((margin, margin),
                        (margin, margin), (0, 0)), mode='edge')
    tf doesn't have edge repeat mode,, so doing it with tile
    Assumes image has 3 channels!!
    """

    num_channels = image.shape[2]

    def repeat_col(col, num_repeat):
        # col is N x 3, ravels
        # i.e. to N*3 and repeats, then put it back to num_repeat x N x 3
        with tf.name_scope('repeat_col'):
            return tf.reshape(
                tf.tile(tf.reshape(col, [-1]), [num_repeat]),
                [num_repeat, -1, num_channels])

    with tf.name_scope('pad_image_edge'):
        top = repeat_col(image[0, :, :], margin)
        bottom = repeat_col(image[-1, :, :], margin)

        image = tf.concat([top, image, bottom], 0)
        # Left requires another permute bc how img[:, 0, :]->(h, 3)
        left = tf.transpose(repeat_col(image[:, 0, :], margin), perm=[1, 0, 2])
        right = tf.transpose(
            repeat_col(image[:, -1, :], margin), perm=[1, 0, 2])
        image = tf.concat([left, image, right], 1)

        return image


def random_flip(image, seg_gt, kp):
    """
    mirrors image L/R and kp
    """

    uniform_random = tf.random.uniform([], 0, 1.0)
    mirror_cond = tf.less(uniform_random, .5)

    new_image, new_gt, new_kp = tf.cond(mirror_cond, lambda: flip_image(image, seg_gt, kp),
                                lambda: (image, seg_gt, kp))
    return new_image, new_gt, new_kp


def flip_image(image, seg_gt, kp):
    """
    Flipping image and kp.
    kp is 3 x N!
    """
    image = tf.reverse(image, [1])
    seg_gt = tf.reverse(seg_gt, [1])
    new_kp = kp

    new_x = tf.cast(tf.shape(image)[0], dtype=kp.dtype) - kp[0, :] - 1
    new_kp = tf.concat([tf.expand_dims(new_x, 0), kp[1:, :]], 0)
    # Swap left and right limbs by gathering them in the right order
    # For COCO+
    swap_inds = tf.constant(
        [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 16, 15, 18, 17])
    new_kp = tf.transpose(tf.gather(tf.transpose(new_kp), swap_inds))

    return image, seg_gt, new_kp
