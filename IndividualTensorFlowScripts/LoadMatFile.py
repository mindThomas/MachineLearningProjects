# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 19:06:00 2017

@author: Thomas
"""

#%%
import numpy as np

import scipy.io
mat = scipy.io.loadmat('mnist_all.mat')

print("MAT file loaded. Contains", len(mat), "datasets. Example size:", mat['train1'].shape)

scipy.io.savemat('test.mat', mat)