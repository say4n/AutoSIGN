# Include Tesseract and other libraries for 
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')

import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *  # Inception Block Models

#%matplotlib inline
#%load_ext autoreload
#%autoreload 2

allowed_dist = 0.7

def triplet_loss(y_true, y_pred, alpha = 0.2):

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # Step 1: Compute the (encoding) distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))
    # Step 2: Compute the (encoding) distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.maximum(tf.reduce_mean(basic_loss), 0.0)
    
    return loss

def verify(image_path, identity, database, model):
  
    encoding = img_to_encoding(image_path, model)
    dist = np.linalg.norm(encoding-database[identity])
    
    if dist < allowed_dist:
        print("It's " + str(identity) + ", welcome home!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False

        
    return dist, door_open

#np.set_printoptions(threshold=np.nan)

AutoSIGNmodel = AutoSIGNModel(input_shape=(3, 96, 96))

database = {}
database["sign1"] = img_to_encoding("images/sign1.jpg", AutoSIGNmodel)

load_weights_from_FaceNet(AutoSIGNmodel)

AutoSIGNmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])

verify("images/sign2.jpg", "sign1", database, AutoSIGNmodel)