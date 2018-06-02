# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 01:13:02 2018

@author: Fakrul-IslamTUSHAR
"""

##Importing the libarary
from __future__ import absolute_import, division, print_function
import tflearn
import numpy as np
import tensorflow as tf
from tflearn.data_utils import to_categorical
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from tflearn.layers.normalization import local_response_normalization

# =============================================================================
# Loading data
# =============================================================================
x,y=tflearn.data_utils.load_csv ('hw2data.csv',has_header=False,categorical_labels=False) #load data

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
Y=tflearn.data_utils.to_categorical (y, 2) #one-hot encoding.
X,Y=shuffle(x,Y,random_state=2) #suffle tha data
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=2) #spliting in training and testing 

# =============================================================================
# Now define the model
# =============================================================================

with tf.Graph().as_default():

     input_layer = tflearn.input_data(shape=[None, 10], name='input')
     dense1 = tflearn.fully_connected(input_layer, 64, activation='relu',weights_init=tflearn.initializations.zeros(),bias_init=tflearn.initializations.zeros(),regularizer='L2')
     network = tflearn.layers.normalization.batch_normalization(dense1)
     dense2 = tflearn.fully_connected(network, 64, activation='relu',weights_init=tflearn.initializations.zeros(),bias_init=tflearn.initializations.zeros(),regularizer='L2')
     softmax = tflearn.fully_connected(dense2, 2, activation='softmax')
     regression = tflearn.regression(softmax, optimizer='adam',
                                learning_rate=0.003,
                                loss='categorical_crossentropy')

     model = tflearn.DNN(regression,tensorboard_verbose=3)

     model.fit(x_train,y_train, show_metric=True, batch_size=64, n_epoch=300, snapshot_epoch=True,validation_set=(x_test,y_test))