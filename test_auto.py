import tensorflow as tf
from src.ops import keypoint_l1_loss, mesh_reprojection_loss
import numpy as np
tf.summary.trace_on(graph=True, profiler=False)
generator_kp_loss_weight = 0.4
batch_size = 1


@tf.function
def function_1():
    a = 5
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! NOOOOOOOOOOOOOOOOOOOOOOOOO  !!!!!!!!!!!!!!!!!!!!!!!")
    tf.print("autographed")

    b = []
    small_kps = tf.ones((10,20,3))
    pred_kps = tf.ones((10,20,2))
    small_seg = tf.ones((1,6890,2))
    silhouette_gt = tf.ones((300,2))
    silhouette_gt= tf.concat([tf.zeros((300,1)), silhouette_gt], axis=1)

    for i in tf.range(3):
        b.append(generator_kp_loss_weight * keypoint_l1_loss(small_kps, pred_kps)) 
        b.append(generator_kp_loss_weight * mesh_reprojection_loss(silhouette_gt, small_seg,
                                                                   batch_size))

    return a

a = tf.constant(2, tf.float32)
tf.debugging.check_numerics(a, "oh no")

b = tf.constant(np.inf, tf.float32)
try:
    tf.debugging.check_numerics(b, "oh no")
except:
    print("nothing")
print("hello")

function_1()
function_1()
writer = tf.summary.create_file_writer('simple_logs/auto_test2')
with writer.as_default():
  tf.summary.trace_export(
      name="my_func_trace",
      step=0)

