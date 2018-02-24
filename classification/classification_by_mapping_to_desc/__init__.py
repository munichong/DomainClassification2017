import tensorflow as tf
w = tf.Variable(tf.random_uniform([], minval=0.0, maxval=1.0))
w1 = tf.subtract(tf.constant(1.0), w)
x = tf.constant([[1.0, 1.5, 2.0], [2.0, 1.5, 1.0]])

s = tf.reshape(tf.reduce_sum(x, axis=1), (-1, 1))
a = tf.divide(x, s)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    init.run()
    print(s.eval())
    print(type(a.eval()))