import numpy as np
import random
import tensorflow as tf
import time
import os

HIDDEN_UNITS_1 = 64
HIDDEN_UNITS_2 = 64


class TRPO(object):
    def __init__(self, state_space, action_space, max_episode_num, episode_lens, discount_factor=0.95,
                 actor_learning_rate=1e-3, critic_learning_rate=1e-3, mini_batch_size=64, epochs=10):
        self.state_space = state_space
        self.action_space = action_space
        self.max_episode_num = max_episode_num
        self.episode_lens = episode_lens
        self.discount_factor = discount_factor
        self.a_lr = actor_learning_rate
        self.c_lr = critic_learning_rate
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs

        self.sess = tf.Session()
        self.hidden_units_1 = HIDDEN_UNITS_1
        self.hidden_units_2 = HIDDEN_UNITS_2

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=1)

    # create action network
    def create_actor_network(self, name, state, trainable=True):
        with tf.variable_scope(name):
            # two hidden layer
            l1 = tf.layers.dense(state, self.hidden_units_1, tf.nn.relu, trainable=trainable)
            l2 = tf.layers.dense(l1, self.hidden_units_2, tf.nn.relu, trainable=trainable)
            mu = 2*tf.layers.dense(l2, self.action_space, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l2, self.action_space, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    # create value network
    def create_critic_network(self, name, state, trainable=True):
        with tf.variable_scope(name):
            # built value network
            l1 = tf.layers.dense(state, self.hidden_units_1, tf.nn.relu, trainable=trainable)
            l2 = tf.layers.dense(l1, self.hidden_units_2, tf.nn.relu, trainable=trainable)
            value = tf.layers.dense(l2, 1)
        return value

    def train_step_generate(self):
        