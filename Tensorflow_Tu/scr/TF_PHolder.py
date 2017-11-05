import tensorflow as tf
import numpy as np

rand_array = np.random.rand(4,4)

x = tf.placeholder(tf.float32, shape = (4,4), name = 'x')
y = tf.matmul(x,x)

with tf.Session() as sess:
    print(rand_array)
    for _ in range(10):
        yr = sess.run(y, feed_dict = {x:rand_array})
        print(yr)
