#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 12:52:51 2018

@author: francesco
"""

import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical
from tflearn.data_preprocessing import DataPreprocessing
# Other import possibly needed
from __future__ import absolute_import, division, print_function
import numpy as np
from tflearn.data_utils import to_categorical
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from tflearn.layers.normalization import local_response_normalization



with tf.Graph().as_default():
    # Normalize the data
    #
    # Possible data normalizations Section
    #

    # Structure of the network
    #
    # input layer, hidden layers, output layer, possible dropout and batch normalization layers
    # Optimizer choice, regression
    #
    input_layer = tflearn.input_data(shape=[None, 10], name='input')
    dense1 = tflearn.fully_connected(input_layer, 64, activation='relu',weights_init=tflearn.initializations.xavier(),bias_init=tflearn.initializations.xavier(),regularizer='L2')
    network = tflearn.layers.normalization.batch_normalization(dense1)
    dense2 = tflearn.fully_connected(network, 64, activation='relu',weights_init=tflearn.initializations.xavier(),bias_init=tflearn.initializations.xavier(),regularizer='L2')
    softmax = tflearn.fully_connected(dense2, 2, activation='softmax')
    regression = tflearn.regression(softmax, optimizer='adam',
                                learning_rate=0.003,
                                loss='categorical_crossentropy')
    lm = tflearn.DNN(regression,tensorboard_verbose=0)
   
    lm.load("mymodel.tfl")
    
 
print("Accuracy: {}%".format(100 * np.mean(testlab == np.argmax(lm.predict(testfv), axis=1))))