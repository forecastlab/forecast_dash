#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This code originates from the M4 Competition NN Benchmarks.
# https://github.com/Mcompetitions/M4-methods/blob/master/ML_benchmarks.py

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.optimizers import rmsprop

def create_inputs(y, input_size = 3):
    """
    Creates multiple points to be used as inputs for each step. 
    :param y: an individual time series
    :param input_size: number of input points for the forecast
    :return:
    """

    x_train, y_train = y[:-1], np.roll(y, -input_size)[:-input_size]
    x_test = np.array( y[-input_size:] ).T[0]

    # reshape input to be [samples, time steps, features] (N-NF samples, 1 time step, 1 feature)
    temp_train = np.roll(x_train, -1)

    for x in range(1, input_size):
        x_train = np.concatenate((x_train[:-1], temp_train[:-1]), 1)
        temp_train = np.roll(temp_train, -1)[:-1]

    return x_train, y_train, x_test

def rnn_bench(y, h = 10, input_size = 3, level = (80,95)):
    """
    Forecasts using 6 SimpleRNN nodes in the hidden layer and a Dense output layer
    :param y: an individual time series
    :param h: forecasting horizon
    :param input_size: number of points used as input
    :return:
    """
    
    # get inputs 
    x_train, y_train, x_test = create_inputs(y, input_size)
    
    # reshape to match expected input
    x_train = np.reshape(x_train, (-1, input_size, 1))
    x_test = np.reshape(x_test, (-1, input_size, 1))

    # create the model
    model = Sequential([
        SimpleRNN(6, input_shape=(input_size, 1), activation='linear',
                  use_bias=False, kernel_initializer='glorot_uniform',
                  recurrent_initializer='orthogonal', bias_initializer='zeros',
                  dropout=0.0, recurrent_dropout=0.0),
        Dense(1, use_bias=True, activation='linear')
    ])
    opt = rmsprop(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=opt)

    # fit the model to the training data
    model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=0)

    # make predictions
    y_hat = []
    last_prediction = float(model.predict(x_test)[0])
    for i in range(0, h):
        y_hat.append(last_prediction)
        x_test[0] = np.roll(x_test[0], -1)
        x_test[0, (len(x_test[0]) - 1)] = last_prediction
        last_prediction = float(model.predict(x_test)[0])
    
    return {'method':'RNN Benchmark', 'mean': y_hat, 'upper': y_hat, 'lower': y_hat}

