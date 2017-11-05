import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 64
    
    fig, axes = plt.subplots(8, 8)
    fig.subplots_adjust(hspace=0.3,wspace=0.3)
    
    for i,ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
         
        if cls_pred is None:
            xlabel = 'True: {0}'.format(cls_true[i])
        else:
            xlabel = 'True: {0},Pred: {1}'.format(cls_true[i], cls_pred[i])
             
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])


data = input_data.read_data_sets('./MINST/', one_hot=True)
x_batch, y_true_batch = data.train.next_batch(64)
data.test.cls = np.array([label.argmax() for label in y_true_batch])

img_size = 28
img_size_flat = img_size*img_size
img_shape = (img_size, img_size)
num_classes = 10
weights = tf.Variable(tf.zeros([img_size_flat,num_classes]))
biases = tf.Variable(tf.zeros([num_classes]))

saver = tf.train.Saver()

result = tf.matmul(x_batch,weights)+biases
with tf.Session() as sess:
    saver.restore(sess, "./model/simpleRegr/model.ckpt")
    pred = sess.run(result)
    print(pred)
    predarg = np.array([label.argmax() for label in pred])
    print(data.test.cls)
    print(predarg)
    images = x_batch
    cls_true = data.test.cls
    plot_images(images = images, cls_true = cls_true, cls_pred=predarg)
    plt.show()
    


