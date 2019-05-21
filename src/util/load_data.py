import numpy as np
import skimage.io as io
import tensorflow as tf

def read_and_decode(filename_queue):
    
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
        'mask_raw': tf.FixedLenFeature([], tf.string)
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    annotation = tf.decode_raw(features['mask_raw'], tf.uint8)
    
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    
    image_shape = ([height, width, 3])
    annotation_shape = ([height, width, 1])
    
    image = tf.reshape(image, image_shape)
    annotation = tf.reshape(annotation, annotation_shape)
 
    return image, annotation

def example_run():
    tfrecords_filename = 'simple_datasets/lsp_few.tfrecords'
    filename_queue = tf.train.string_input_producer(
        [tfrecords_filename], num_epochs=10)

    # Even when reading in multiple threads, share the filename
    # queue.
    image, annotation = read_and_decode(filename_queue)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session()  as sess:

        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Let's read off 3 batches just for example

        # We selected the batch size of two
        # So we should get two image pairs in each batch
        # Let's make sure it is random
        for i in range(8):
            img, anno = sess.run([image, annotation])
            print(img.shape)
            print(anno.shape)
            print('current batch')

            io.imshow(img)
            io.show()

            io.imshow(np.squeeze(anno))
            io.show()


        coord.request_stop()
        coord.join(threads)
