# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 11:19:13 2017

@author: Thomas
"""

import numpy as np
import scipy.io

import tensorflow as tf
from keras.utils import np_utils

def generate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):        
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') # stride of 1

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')   # pooling over 2x2 blocks with a stride of 1

#%% Load dataset
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

trainData = mnist.data[:60000].astype(float) / 255.
trainLabels = mnist.target[:60000]
testData = mnist.data[60000:].astype(float) / 255.
testLabels = mnist.target[60000:]

# Create the model
tf.reset_default_graph() # reset if we are rerunning code to avoid variable re-use

# Input layer
x = tf.placeholder(tf.float32, [None, 784]) 

# Reshape input to [batchsize x width x height x color channels]
x_image = tf.reshape(x, [-1, 28, 28, 1])

# First convolutional layer with 32x 5x5 Kernels
# Using ReLU activation function
W_conv1 = weight_variable([5, 5, 1, 32]) # [patchX x patchY x input channels x output channels]
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1) # reduces output to 14x14x32

# Second convolutional layer with 2x 5x5 Kernels
# Using ReLU activation function
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2) # reduces output to 7x7x64

# Fully connected layer taking the 7x7x64 output down to a vector of 1024 elements
# Using ReLU activation function
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) # reshape pooling layer output into a batch of vectors (if multiple inputs are fed into the network in a batch)
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout layer just before readout layer to help against overfitting
keep_prob = tf.placeholder(tf.float32) # placeholder used to be able to enable and disable dropout
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer provides the final layer with the number of classes as output
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Define placeholder for true one-hot label
y_ = tf.placeholder(tf.float32, [None, 10])

# Softmax on output layer + Cross entropy for the cost function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
#optimizer = tf.train.GradientDescentOptimizer(0.05)
#optimizer = tf.train.RMSPropOptimizer(learning_rate=0.05, momentum=0.9, epsilon=1e-08, decay=0.0)

train_step = optimizer.minimize(cost)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()  # sess.run(tf.global_variables_initializer())

#saver.restore(sess, "./mnist_tensorflow_cnn.ckpt")

#%% Perform classification with MLP using Keras
trainLabels_01_tf = tf.one_hot(trainLabels, 10) # 10 different classes
testLabels_01_tf = tf.one_hot(testLabels, 10)

#trainLabels_01 = trainLabels_01.eval() # requires than an interactive session has already been started
trainLabels_01 = tf.Session().run(trainLabels_01_tf)
testLabels_01 = tf.Session().run(testLabels_01_tf)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy_tf = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

i = 0
for iteration in range(16):
    g = generate_minibatches(trainData, trainLabels_01, 50, True)      
    for batch in g:        
        batch_xs, batch_ys = batch   
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})        
  
        if i % 100 == 0:
            accuracy = accuracy_tf.eval(feed_dict={x: testData, y_: testLabels_01, keep_prob: 1.0})                  
            print("Accuracy: %f" % (accuracy))    
      
        i += 1
        
#%% Test trained model        
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy_tf = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

writer = tf.summary.FileWriter('log', sess.graph) # store graph for visualization with "tensorboard --logdir=log"
accuracy = sess.run(accuracy_tf, feed_dict={x: testData, y_: testLabels_01, keep_prob: 1.0})
writer.close()

print("Accuracy: %f" % accuracy)    

save_path = saver.save(sess, "./mnist_tensorflow_cnn.ckpt")