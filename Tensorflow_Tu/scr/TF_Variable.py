import tensorflow as tf

with tf.device("/cpu:0"):
    biases = tf.Variable([20,3], name = 'biases')
biases_2 = tf.Variable(tf.zeros([2]), name = 'biases')
biases_20 = tf.Variable(tf.zeros([2,]), name = 'biases')
biases_23 = tf.Variable(tf.zeros([2,3]), name = 'biases')

weight = tf.Variable(biases.initialized_value(),name = 'weight')

init_ops = tf.global_variables_initializer()
v_assign =  biases.assign([4,5])

saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(init_ops)
    save_path = saver.save(sess, "./model/VarModel/model.ckpt")
    print("Model saved in file: %s" % save_path)
    print(weight)
    print(weight.eval())
    print(biases)
    print(biases.eval())
    print(biases_2)
    print(biases_2.eval())
    print(biases_20)
    print(biases_20.eval())
    print(biases_23)
    print(biases_23.eval())
    sess.run(v_assign)
    print('#########assign########')
    print(biases)
    print(biases.eval())
    print('#########value########')
    print(biases_23.value())
    print('#########get_shape########')
    print(biases_23.get_shape())
    print('#########read_value########')
    print(biases_23.read_value())
    print('#######device#######')
    print(biases_23.device)
    print('#######graph#######')    
    print(biases_23.graph)
    print('#######op#######')
    print(biases_23.op)
    print('#######name#######')
    print(biases_23.name)