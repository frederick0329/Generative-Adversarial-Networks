from layers import *
import tensorflow as tf

class Discriminator():
    def build_network(self, image, image_size, hidden_size):
        with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
            out = fully_connected('fc1', image, image_size, hidden_size)
            out = tf.nn.leaky_relu(out)
            out = fully_connected('fc2', out, hidden_size, hidden_size)
            out = tf.nn.leaky_relu(out)
            out = fully_connected('fc3', out, hidden_size, 1)
            return out

class Generator():
    def build_network(self, noise, latent_size, image_size, hidden_size):
        with tf.variable_scope('Generator'):
            out = fully_connected('fc1', noise, latent_size, hidden_size)
            out = tf.nn.relu(out)
            out = fully_connected('fc2', out, hidden_size, hidden_size)
            out = tf.nn.relu(out)
            out = fully_connected('fc3', out, hidden_size, image_size)
            out = tf.nn.tanh(out)
            return out
