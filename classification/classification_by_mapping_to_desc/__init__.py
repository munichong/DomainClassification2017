import tensorflow as tf
import numpy as np

tf.InteractiveSession()

x1 = tf.constant([[1.0, 1.5, 2.0], [2.0, 1.5, 1.0]])
x2 = tf.constant([[2.0, 2.5, 3.0], [3.0, 2.5, 3.0]])

x12product = tf.multiply(x1, x2).eval()

x12 = tf.stack([x1, x2, x12product], axis=-1)  # [None， target， 3]

# x12 = tf.concat([x12, x12product], axis=-1)

print(x12.eval())

# w = tf.truncated_normal([3, 2], stddev=0.1)  # [2, target]
w = tf.constant([[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]]])
print(w.get_shape())
x12w = tf.multiply(x12, w)
print(x12w.eval())
print(tf.reduce_sum(x12w, axis=-1).eval())



# a = tf.truncated_normal([2, 2, 3], stddev=0.1)
# b = tf.truncated_normal([2, 3, 2], stddev=0.1)
# c = tf.matmul(a, b)
# print(c.eval())

# def some_function(tensor):
#   return tf.reduce_sum(tensor, -1)
#
# a = tf.stack([x1, x2], axis=1)
# d = tf.map_fn(some_function, a, dtype=tf.float32)

