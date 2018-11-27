import numpy as np
import tensorflow as tf

np_a = np.ones([2,2,2,2])
tf_a = tf.convert_to_tensor(np_a)
tf_q, tf_r = tf.qr(np_a)

with tf.Session():
    np_q = tf_q.eval()
    np_r = tf_r.eval()

print(np_a)
print(np_q)
print(np_r)