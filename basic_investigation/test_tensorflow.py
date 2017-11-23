import tensorflow as tf


for _ in range(3):
    a = tf.Variable(3)
    b = tf.Variable(2)
    c = tf.assign(a, a + b)

with tf.Session() as sess:
    a.initializer.run()
    b.initializer.run()
    sess.run(c)
    print(a.eval())