# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 15:01:37 2018

@author: Fakrul-IslamTUSHAR
"""

from __future__ import absolute_import, division, print_function
import tflearn
import numpy as np
import tensorflow as tf
from tflearn.data_utils import to_categorical
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

# =============================================================================
# Loading data
# =============================================================================
x,y=tflearn.data_utils.load_csv ('hw2data.csv',has_header=False,categorical_labels=False)

#masking the y=-1 to 0.
for i in range(len(y)): 
    if (y[i]=='-1'):
        y[i]=0
    else:
        y[i]=1

print(y[5000]) #printing the value of 5000 number lavel.
# =============================================================================
# Data Suffle and Splitting betweentraing and testing set.
# =============================================================================
Y=tflearn.data_utils.to_categorical (y, 2)
X,Y=shuffle(x,Y,random_state=2)
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=2)

# =============================================================================
# Now define the model
# =============================================================================
with tf.Graph().as_default():

    input_layer = tflearn.input_data(shape=[None, 10], name='input')
    dense1 = tflearn.fully_connected(input_layer,64, activation='relu',weights_init=tflearn.initializations.uniform(),bias_init=tflearn.initializations.uniform(),regularizer='L2')
    #batch_normalization = tflearn.layers.normalization.batch_normalization(dense1)
    dense2 = tflearn.fully_connected(dense1, 64, activation='relu',weights_init=tflearn.initializations.uniform(),bias_init=tflearn.initializations.uniform(),regularizer='L2')
    softmax = tflearn.fully_connected(dense2, 2, activation='softmax')
    regression = tflearn.regression(softmax, optimizer='adam',
                                learning_rate=0.003,
                                loss='categorical_crossentropy')

    model = tflearn.DNN(regression,tensorboard_verbose=0)
    model.fit(x_train,y_train, show_metric=True, batch_size=64, n_epoch=300, snapshot_epoch=False,validation_set=(x_test,y_test))