# NORMAL FUNCTION IMPORTS

import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from collections import deque
from tqdm import tqdm
import pylab
import csv

# TENSORFLOW Imports 

import keras
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Add, Lambda, TimeDistributed, LSTM, concatenate
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.layers import GaussianNoise


# ==============================================================================
# -- Network Details for Training ----------------------------------------------
# Network_LSTM: This scripts contains following functions
#         >> High level network (Meta Model)
#         >> Low level network (Model)
# ==============================================================================

class Network_Model():

    def meta_model(input_shape, goal_space):
        x_input = (Input(input_shape))
        x = x_input
        x = TimeDistributed(GaussianNoise(0.1))(x)
        x = LSTM(512, input_shape = input_shape, activation = "tanh", kernel_initializer = 'he_uniform')(x)
        x = Dense(256, activation = "relu", kernel_initializer = 'he_uniform')(x)
        x = Dense(64, activation = "relu", kernel_initializer = 'he_uniform')(x)
        x = Dense(goal_space, activation = "relu", kernel_initializer = 'he_uniform')(x)
        model = Model(inputs = x_input, outputs = x, name = 'Meta_Goal')
        model.compile(loss = "mse", optimizer = RMSprop(lr = 0.00025, rho = 0.95, epsilon = 0.01), metrics =["accuracy"])
        model.summary()
        return model

    def controller_model(goal_shape, input_shape, action_space):

        # Goal input
        inputA = (Input(goal_shape))    
        x= Dense(512, input_shape = goal_shape, activation ="relu", kernel_initializer = 'he_uniform')(inputA)
        #x = Model(inputs = inputA, outputs = x)

        # History input
        inputB = (Input(input_shape))
        y = TimeDistributed(GaussianNoise(0.1))(inputB)
        y = LSTM(512, input_shape = input_shape, activation = "tanh", kernel_initializer = 'he_uniform')(y)
        #y = Model(inputs = inputB, outputs = y)

        combined = concatenate([x, y])

        z = Dense(256, activation = "relu", kernel_initializer = 'he_uniform')(combined)
        z = Dense(64, activation = "relu", kernel_initializer = 'he_uniform')(z)
        z = Dense(action_space, activation = "linear", kernel_initializer = 'he_uniform')(z)
        model = Model(inputs = [inputA, inputB], outputs = z, name = 'Lane_Change')
        model.compile(loss = "mse", optimizer = RMSprop(lr = 0.00025, rho = 0.95, epsilon = 0.01), metrics = ["accuracy"])
        model.summary()
        return model

