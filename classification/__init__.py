import tensorflow as tf
import numpy as np


t = tf.constant([[0.6,0.2,0.3], [0.5,0.4,0.7]])
y = tf.placeholder(dtype=tf.int32, shape=[1, None])

# one_hot_mask = tf.one_hot(y, t.shape[1], on_value = True, off_value = False, dtype = tf.bool)
# # a = tf.boolean_mask(t, one_hot_mask)

ranked = tf.nn.top_k(t, k=3).indices
y_2d = tf.reshape(y, (y.shape[-1], 1))
rank_sum = tf.reduce_sum(tf.where(tf.equal(ranked, y_2d))[:, -1])

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    print(sess.run(rank_sum, feed_dict={y: np.array([[1,2]])}))

'''
# m = tf.constant([[0, 2, 1],[2, 0, 1]])  # matrix
y = tf.placeholder(tf.int32, shape=[None], name='target')
# print(tf.shape(y))
# a = tf.shape(y)[0]
# t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
# print(tf.shape(t))  # [2, 2, 3]

# yt = tf.squeeze(y)
yt = tf.expand_dims(y, 0)
yt = tf.reshape(y, shape=[tf.shape(y)[1], 1])

#
# ranked = tf.where(tf.equal(m, y))[:,-1]


init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    print(sess.run(yt, feed_dict={y:np.array([1,2,3])}))



print(np.array([1,2,3]).shape)

input = tf.placeholder(dtype=tf.float32, shape=[None])
# print(input.shape)
# input = tf.expand_dims(input, 0)
# print(input.shape)

input_flattened = tf.reshape(input, shape=[tf.shape(input)[0], 1])

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    input_val, input_flattened_val =sess.run([input, input_flattened], feed_dict={input: np.array([[1,2,3]])})
    print(input_val.shape)
    print(input_flattened_val.shape)
'''