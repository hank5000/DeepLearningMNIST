from __future__ import print_function
import tensorflow as tf
import numpy as np

class MNISTDetector:
    def weight_variable(self,shape, elements, Wname):
        return tf.Variable(np.arange(elements).reshape(shape), dtype=tf.float32, name=Wname)

    def bias_variable(self,shape, elements, Bname):
        return tf.Variable(np.arange(elements).reshape(shape), dtype=tf.float32, name=Bname)

    def __init__(self):
        # define placeholder for inputs to network
        self.xs = tf.placeholder(tf.float32, [784], name='xs')
        self.ys = tf.placeholder(tf.float32, shape=[None, 10], name='ys')
        self.x_image = tf.reshape(self.xs, [-1, 28, 28, 1], name='xs_new_shape')

        ## conv1 layer ##
        self.W_conv1 = self.weight_variable([5, 5, 1, 32], 5*5*1*32, 'ConvW1') # patch 5x5, in size 1, out size 32
        self.b_conv1 = self.bias_variable( [32], 1*32, 'Convb1')
        self.h_conv1 = tf.nn.relu((tf.nn.conv2d(self.x_image, self.W_conv1, strides=[1, 1, 1, 1], padding='SAME')) + self.b_conv1) # size: 28*28*32
        self.h_pool1 = tf.nn.max_pool(self.h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')            # size: 14*14*32

        ## conv2 layer ##
        self.W_conv2 = self.weight_variable( [5, 5, 32, 64], 5*5*32*64, 'ConvW2') # patch 5x5, in size 32, out size 64
        self.b_conv2 = self.bias_variable( [64], 1*64, 'Convb2')
        self.h_conv2 = tf.nn.relu(tf.nn.conv2d(self.h_pool1, self.W_conv2, strides=[1, 1, 1, 1], padding='SAME') + self.b_conv2) # size: 14*14*64
        self.h_pool2 = tf.nn.max_pool(self.h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')          # size: 7*7*64
        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7*7*64]) # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]

        ## fully connected layer 1 ##
        self.W_fu1 = self.weight_variable( [7*7*64, 1024], 7*7*64*1024, 'FullyW1')
        self.b_fu1 = self.bias_variable( [1024], 1*1024, 'Fullyb1')
        self.h_fu1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fu1) + self.b_fu1)

        ## fully connected layer 2 ##
        self.W_fu2 = self.weight_variable( [1024, 10], 1024*10, 'FullyW2')
        self.b_fu2 = self.bias_variable( [10], 1*10, 'Fullyb2')
        self.prediction = tf.nn.softmax(tf.matmul(self.h_fu1, self.W_fu2) + self.b_fu2)

        # # the error between prediction and real data
        # cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
        # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, "MyNets/save_net.ckpt")

    def detect(self, image):
        prediction = self.sess.run(self.prediction, feed_dict={self.xs:image})
        return prediction


