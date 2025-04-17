import tensorflow as tf
import tensorflow.keras as keras
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
import numpy as np
from random import randint, shuffle, uniform
from Calcul import G1, G2, J
from keras.layers.advanced_activations import PReLU
from time import time


def agent(state_shape, action_shape):
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, input_shape=state_shape, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    return model


test = agent((5,), (2,))
