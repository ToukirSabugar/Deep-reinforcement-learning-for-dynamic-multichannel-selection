from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np 
import tensorflow as tf 
import pandas as pd 
from collections import deque


class channel_env:
    def __init__(self, n_channels):
        self.n_channels = n_channels
        self.reward = 1
        self.action_set = np.arange(n_channels)
        self.action = -1
        self.observation = -1


class QNetwork(tf.keras.Model):
    def __init__(self, learning_rate, state_size, action_size, hidden_size, name="Channel_QNetwork"):
        super(QNetwork, self).__init__(name=name)
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.action_size = action_size
        
        self.w1 = tf.Variable(tf.random.uniform([state_size, hidden_size]))
        self.b1 = tf.Variable(tf.constant(0.01, shape=[hidden_size]))

        self.w2 = tf.Variable(tf.random.uniform([hidden_size, hidden_size]))
        self.b2 = tf.Variable(tf.constant(0.01, shape=[hidden_size]))

        self.w_outlayer = tf.Variable(tf.random.uniform([hidden_size, action_size]))
        self.b_outlayer = tf.Variable(tf.random.uniform([action_size]))

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)  # Cast inputs to float32
        h1 = tf.matmul(inputs, self.w1) + self.b1
        h1 = tf.nn.relu(h1)

        h2 = tf.matmul(h1, self.w2) + self.b2
        h2 = tf.nn.relu(h2)

        out_layer = tf.matmul(h2, self.w_outlayer) + self.b_outlayer
        return out_layer

    def loss_fn(self, states, actuals, actions):
        action_onehot_vec = tf.one_hot(actions, self.action_size)
        Q_pred = tf.reduce_sum(tf.multiply(self(states), action_onehot_vec), axis=1)
        Q_pred = tf.cast(Q_pred, tf.float64)  # Cast Q_pred to float64
        return tf.reduce_mean(tf.square(actuals - Q_pred))


    @tf.function
    def train_step(self, states, actuals, actions, optimizer):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(states, actuals, actions)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    
class ExpMemory():
    def __init__(self, in_size):
        self.buffer_in = deque(maxlen=in_size)

    def add(self, exp):
        self.buffer_in.append(exp)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer_in)), size=batch_size, replace=False)
        res = []
        for i in idx:
            res.append(self.buffer_in[i])
        return res

