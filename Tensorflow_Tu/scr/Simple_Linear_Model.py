import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('./MINST/', one_hot=True)

print('Size of: ')
print('- Training-set:\t\t{}'.format(len(data.train.labels)))
print('- Test-set:\t\t{}'.format(len(data.test.labels)))
print('- Validation-set:\t{}'.format(len(data.validation.labels)))

data.test.cls = np.array([label.argmax() for label in data.test.labels])

img_size = 28
img_size_flat = img_size*img_size
img_shape = (img_size, img_size)
num_classes = 10

x = tf.placeholder(tf.float32, [None, img_size_flat])
y_true = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.placeholder(tf.int64, [None])

weights = tf.Variable(tf.zeros([img_size_flat,num_classes]))
biases = tf.Variable(tf.zeros([num_classes]))

logits = tf.matmul(x,weights) + biases
y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred, dimension=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y_true, 
                                                        logits=logits)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

correct_prediction = tf.equal(y_true_cls, y_pred_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


    
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    fig, axes = plt.subplots(3, 3)
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
        
def plot_weights():
    w = session.run(weights)
    w_min = np.min(w)
    w_max = np.max(w)
    fig,axes = plt.subplots(3,4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        if i<10:
            image = w[:,i].reshape(img_shape)
            ax.set_xlabel('Weight: {0}'.format(i))
            ax.imshow(image, vmin = w_min,vmax = w_max,cmap='seismic')
    ax.set_xticks([])
    ax.set_yticks([])
    
def plot_errors():
    correct, cls_pred = session.run([correct_prediction, y_pred_cls],
                                    feed_dict = feed_dict_test)

    incorrect = (correct == False)
    images = data.test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = data.test.cls[incorrect]
    plot_images(images=images[0:9], cls_true=cls_true[0:9], cls_pred = cls_pred[0:9])

session = tf.Session()
session.run(tf.global_variables_initializer())
batch_size = 100
for _ in range(100):
    x_batch, y_true_batch = data.train.next_batch(batch_size)
    feed_dict_train = {x:x_batch,
                       y_true:y_true_batch}
    session.run(optimizer, feed_dict = feed_dict_train)
    feed_dict_test = {x: data.test.images,
                      y_true: data.test.labels,
                      y_true_cls: data.test.cls}
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    print("Accuracy on test-set: {0:.1%}".format(acc))
plot_weights()

saver = tf.train.Saver()
save_path = saver.save(session, "./model/simpleRegr/model.ckpt")
b = session.run(biases)
print(b)
w = session.run(weights)
print(w[:,9])

images = data.test.images[0:9]
cls_true = data.test.cls[0:9]
plot_images(images = images, cls_true = cls_true)
plot_errors()
plt.show()