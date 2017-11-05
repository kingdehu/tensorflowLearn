import tensorflow as tf

my_variable = tf.get_variable('my_variable', [4,2,3])
my_int_variable = tf.get_variable("my_int_variable", [4, 2, 3],
                                   dtype=tf.int32, 
                                   initializer=tf.zeros_initializer)
other_variable = tf.get_variable("other_variable", 
                                 dtype=tf.int32, 
                                 initializer=tf.constant([23, 42]))

init_ops = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_ops)
    print(my_variable)
    print(my_variable.eval())
    print('########my_init_variable######')
    print(my_int_variable)
    print(my_int_variable.eval())
    print('########other_variable######')
    print(other_variable)
    print(other_variable.eval())
    sess.run(other_variable.assign([3,4]))
    print(other_variable)
    print(other_variable.eval())
    