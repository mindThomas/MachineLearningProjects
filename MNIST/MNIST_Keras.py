# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 11:19:13 2017

@author: Thomas
"""

import numpy as np
import scipy.io

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils


#%% Load dataset
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

trainData = mnist.data[:60000].astype(float) / 255.
trainLabels = mnist.target[:60000]
testData = mnist.data[60000:].astype(float) / 255.
testLabels = mnist.target[60000:]

#%% Perform classification with MLP using Keras
def one_hot_encode_object_array(arr):
    '''One hot encode a numpy array of objects (e.g. strings)'''
    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))

trainLabels_01 = one_hot_encode_object_array(trainLabels)
testLabels_01 = one_hot_encode_object_array(testLabels)

model = Sequential()
#model = keras.models.load_model('mnist_keras.h5')
model.add(Dense(100, input_shape=(784,))) # 32 hidden layer units/neurons
model.add(Activation('tanh'))
model.add(Dense(10))                    # 10 output layer units/neurons
model.add(Activation('softmax'))
sgd = keras.optimizers.SGD(lr=0.05, momentum=0.0, decay=0.0, nesterov=False)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.get_config()

#%% Train the model
model.fit(trainData, trainLabels_01, nb_epoch=100, batch_size=100)

#%%
loss, accuracy = model.evaluate(testData, testLabels_01)
print("\n\nAccuracy: %f" % accuracy)

model.predict(np.array([mnist.data[10,:]]))

c = np.argmax(model.predict(np.array([mnist.data[10,:]])))
print("True class %d vs Predicted class %d\n" % (mnist.target[10], c))


#mnist.target[10] == np.argmax(model.predict(array([mnist.data[10,:]])))

#%%
w = model.get_weights()
for layer in model.layers:
    weights = layer.get_weights()
    print(weights)        

model.save('mnist_keras.h5')