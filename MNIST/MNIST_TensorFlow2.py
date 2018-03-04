# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 11:19:13 2017

@author: Thomas
"""

import numpy as np
import scipy.io

import tensorflow as tf
from keras.utils import np_utils

# Training Parameters
learning_rate = 0.001
num_steps = 2000
batch_size = 100

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units


# Create the neural network
def my_net(x_dict, num_input, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('NeuralNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        #x = tf.cast(x_dict['input'], dtype=tf.float32)
        
        x = tf.reshape(x_dict['input'], shape=[-1,num_input]) 

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(x, 100)
        # Apply Dropout (if is_training is False, dropout is not applied)
        #fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(y1, n_classes)    

    return out


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = my_net(features, num_input, num_classes, dropout, reuse=False,
                            is_training=True)
    logits_test = my_net(features, num_input, num_classes, dropout, reuse=True,
                           is_training=False)

    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

        # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs


#%% Load dataset
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

trainData = mnist.data[:60000].astype(float) / 255.
trainLabels = mnist.target[:60000]
testData = mnist.data[60000:].astype(float) / 255.
testLabels = mnist.target[60000:]

# Create the model
tf.reset_default_graph() # reset if we are rerunning code to avoid variable re-use
sess = tf.InteractiveSession()

# Build the Estimator
model = tf.estimator.Estimator(model_fn)

# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'input': trainData}, y=trainLabels,
    batch_size=batch_size, num_epochs=None, shuffle=True)
# Train the Model
model.train(input_fn, steps=num_steps)

# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'input': testData}, y=testLabels,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])