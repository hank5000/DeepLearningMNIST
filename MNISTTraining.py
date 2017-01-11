from __future__ import print_function
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
#    if pre(v_xs(n)) = v_ys, return true; correct_prediction=[true, true, true, false ...]
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys,1))
#    transform [true, true, true, false ...] to [1, 1, 1, 0], and calculate the mean value
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy))

def weight_variable(shape, Wname):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=Wname)

def bias_variable(shape, Bname):
    return tf.Variable(tf.constant(0.1, shape=shape), name=Bname)

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784], name='xs')
ys = tf.placeholder(tf.float32, [None, 10], name='ys')
x_image = tf.reshape(xs, [-1, 28, 28, 1], name='xs_new_shape')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

## conv1 layer ##
W_conv1 = weight_variable([5, 5, 1, 32], 'ConvW1') # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32], 'Convb1')
h_conv1 = tf.nn.relu((tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')) + b_conv1) # size: 28*28*32
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')            # size: 14*14*32

## conv2 layer ##
W_conv2 = weight_variable([5, 5, 32, 64], 'ConvW2') # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64], 'Convb2')
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2) # size: 14*14*64
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')          # size: 7*7*64
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]

## fully connected layer 1 ##
W_fu1 = weight_variable([7*7*64, 1024], 'FullyW1')
b_fu1 = bias_variable([1024], 'Fullyb1')
h_fu1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fu1) + b_fu1)
h_fu1_dropout = tf.nn.dropout(h_fu1, keep_prob)
## fully connected layer 2 ##
W_fu2 = weight_variable([1024, 10], 'FullyW2')
b_fu2 = bias_variable([10], 'Fullyb2')
prediction = tf.nn.softmax(tf.matmul(h_fu1_dropout, W_fu2) + b_fu2)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        compute_accuracy(mnist.test.images, mnist.test.labels)

saver = tf.train.Saver()
save_path = saver.save(sess, "MyNets/save_net.ckpt")
print("Save to path: ", save_path)