import tensorflow as tf

with tf.device("/cpu:0"):
    biases = tf.Variable([20,30], name = 'biases')

weight = tf.Variable(biases.initialized_value(),name = 'weight')

v_assign =  biases.assign([4,5])

saver = tf.train.Saver()


with tf.Session() as sess:
    saver.restore(sess, "./model/VarModel/model.ckpt")
    print(weight)
    print(weight.eval())
    print(biases)
    print(biases.eval())