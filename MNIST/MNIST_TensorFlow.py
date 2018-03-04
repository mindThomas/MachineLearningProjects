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

# Hidden layer
#W1 = tf.get_variable('W1', [784, 100], initializer=tf.random_normal_initializer())
#W1 = tf.Variable(tf.random_normal([784, 100]))
W1 = tf.Variable(tf.truncated_normal([784, 100], stddev=1.0 / np.sqrt(784)))
b1 = tf.Variable(tf.zeros([100,]))
z1 = tf.matmul(x, W1) + b1
y1 = tf.nn.tanh(z1)


# Output layer  
#W2 = tf.Variable(tf.random_normal([100, 10]))
W2 = tf.Variable(tf.truncated_normal([100, 10], stddev=1.0 / np.sqrt(100)))
b2 = tf.Variable(tf.zeros([10,]))
y2 = tf.matmul(y1, W2) + b2

# Define the output
y = y2 

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])

# Softmax on output layer + Cross entropy for the cost function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.GradientDescentOptimizer(0.05)
#optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
#optimizer = tf.train.RMSPropOptimizer(learning_rate=0.05, momentum=0.9, epsilon=1e-08, decay=0.0)

train_step = optimizer.minimize(cost)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

saver.restore(sess, "./mnist_tensorflow.ckpt")

#%% Perform classification with MLP using Keras
trainLabels_01_tf = tf.one_hot(trainLabels, 10) # 10 different classes
testLabels_01_tf = tf.one_hot(testLabels, 10)

#trainLabels_01 = trainLabels_01.eval() # requires than an interactive session has already been started
trainLabels_01 = tf.Session().run(trainLabels_01_tf)
testLabels_01 = tf.Session().run(testLabels_01_tf)

for iteration in range(100):
    g = generate_minibatches(trainData, trainLabels_01, 100, True)
    for batch in g:        
        batch_xs, batch_ys = batch   
        _, c = sess.run([train_step, cost], feed_dict={x: batch_xs, y_: batch_ys})        
        
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy_tf = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy = sess.run(accuracy_tf, feed_dict={x: testData, y_: testLabels_01})
    print("Accuracy: %f   (Cost: %f)" % (accuracy, c))    

#%% Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy_tf = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

writer = tf.summary.FileWriter('log', sess.graph) # store graph for visualization with "tensorboard --logdir=log"
accuracy = sess.run(accuracy_tf, feed_dict={x: testData, y_: testLabels_01})
writer.close()

print("Accuracy: %f" % accuracy)    

save_path = saver.save(sess, "./mnist_tensorflow.ckpt")