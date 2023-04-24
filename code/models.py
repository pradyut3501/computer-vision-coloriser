"""
Homework 5 - CNNs
CS1430 - Computer Vision
Brown University
"""

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.layers import \
       Conv2D, MaxPool2D, Dropout, Flatten, Dense

import hyperparameters as hp


class YourModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(YourModel, self).__init__()

        self.optimizer = tf.keras.optimizers.SGD()

        self.architecture = [
              Conv2D(32, 3, 1, activation="relu", padding="same"), MaxPool2D(2, padding="same"),
              Conv2D(64, 3, 1, activation="relu", padding="same"), MaxPool2D(2, padding="same"),
              Conv2D(128, 3, 1, activation="relu", padding="same"), MaxPool2D(2, padding="same"),
              Flatten(), Dropout(0.5),
              Dense(128, activation="relu"),Dense(64, activation="relu"), Dense(hp.num_classes, activation="softmax")
        ]

    def call(self, x):
        """ Passes input image through the network. """

        for layer in self.architecture:
            x = layer(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

        return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)