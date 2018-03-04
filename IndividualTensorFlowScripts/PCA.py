# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 19:06:00 2017

@author: Thomas
"""

#%%
import numpy as np
from sklearn.decomposition import PCA

import scipy.io
mat = scipy.io.loadmat('mnist_all.mat')


train0 = mat['train0'].astype(float) / 255.0
train1 = mat['train1'].astype(float) / 255.0
train2 = mat['train2'].astype(float) / 255.0
train3 = mat['train3'].astype(float) / 255.0
train4 = mat['train4'].astype(float) / 255.0
train5 = mat['train5'].astype(float) / 255.0
train6 = mat['train6'].astype(float) / 255.0
train7 = mat['train7'].astype(float) / 255.0
train8 = mat['train8'].astype(float) / 255.0
train9 = mat['train9'].astype(float) / 255.0

dataset = np.concatenate((train0, train1, train2, train3, train4, train5, train6, train7, train8, train9))

if dataset.shape[0] == 60000:
    print("Concatenated successfully\n")

#%% Reduce dimension with PCA

pca = PCA(n_components=30)
pca.fit(dataset)

dataset_reduced = pca.transform(dataset)
dataset_reduced.shape

dataset_reduced_mat = {}
dataset_reduced_mat['dataset'] = dataset_reduced

scipy.io.savemat('datasetPCA.mat', dataset_reduced_mat)

W = pca.components_
W_mat = {}
W_mat['W'] = W
scipy.io.savemat('W.mat', W_mat)

#%% Perform classification with MLP using Keras