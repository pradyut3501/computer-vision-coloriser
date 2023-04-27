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


class CNNModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(CNNModel, self).__init__()

        self.optimizer = tf.keras.optimizers.SGD()

        self.architecture = [
              Conv2D(32, 3, 1, activation="relu", padding="same"), 
              MaxPool2D(2, padding="same")
        ]

    def call(self, x):
        """ Passes input image through the network. """

        for layer in self.architecture:
            x = layer(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """
        #TODO: find new loss function
        return tf.keras.losses.MeanSquaredError(labels, predictions)
    
class GANModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(GANModel, self).__init__()

        self.optimizer = tf.keras.optimizers.SGD()

        self.architecture = [
              Conv2D(32, 3, 1, activation="relu", padding="same"), MaxPool2D(2, padding="same")
        ]

    def call(self, x):
        """ Passes input image through the network. """

        for layer in self.architecture:
            x = layer(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """
        #TODO: find new loss function
        return tf.keras.losses.MeanSquaredError(labels, predictions)