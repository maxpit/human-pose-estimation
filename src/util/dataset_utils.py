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

def write_example(height, width, center, label, img_path, image_data, seg_data):
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
