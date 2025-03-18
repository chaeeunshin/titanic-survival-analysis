#!/usr/bin/env python
# coding: utf-8

import Preprocessing
import tensorflow as tf

NeuralNetworkModel = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation = 'relu', input_shape = [7,]),
    tf.keras.layers.Dense(8, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])
opt = tf.keras.optimizers.Adam(learning_rate = 1e-3)
NeuralNetworkModel.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['acc'])

df = Preprocessing.df 

NeuralNetworkModel.fit(df.x_scaled, df.y, epochs=100, batch_size=32)