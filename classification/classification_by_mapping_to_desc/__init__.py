import tensorflow as tf
w = tf.Variable(tf.random_uniform([], minval=0.0, maxval=1.0))
w1 = tf.subtract(tf.constant(1.0), w)
x = tf.constant([[1.0, 2.0], [2.0, 1.0]])

a = tf.multiply(x, w)
b = tf.multiply(x, w1)
y = tf.add(tf.multiply(x, w), tf.multiply(x, w1))

init = tf.global_variables_initializer()


with tf.Session() as sess:
    init.run()
    print(a.eval())
    print(b.eval())
    print(sess.run(y))
    print(sess.run(y))