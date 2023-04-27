"""
Homework 5 - CNNs
CS1430 - Computer Vision
Brown University
"""

import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf

import hyperparameters as hp
from models import CNNModel, GANModel
from preprocess import Datasets
from skimage.transform import resize

from skimage.io import imread
from matplotlib import pyplot as plt
import numpy as np


def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        default='cnn',
        choices=['cnn', 'gan'],
        help='''Which model to use''')
    parser.add_argument(
        '--data',
        default='TODO',
        help='Location where the dataset is stored.')
    parser.add_argument(
        '--evaluate')
    return parser.parse_args()


def train(model, datasets):
    """ Training routine. """
    model.compile(
        optimizer=model.optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=["mean_squared_error"])

    color, gray = datasets.load_data()
    # color_val = color[:-10000]
    # gray_val = gray[:-10000]
    # color_train = color[-10000:]
    # gray_train = gray[-10000:]
    color_train = color[:100]
    gray_train = gray[:100]

    #print(color_val.shape, gray_val.shape, color_train.shape, gray_train.shape)

    print("Fit model on training data")
    history = model.fit(
        gray_train,
        color_train,
        batch_size=64,
        epochs=50,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        #validation_data=(gray_val, color_val),
    )

    print(history.history)


def test(model, test_data):
    """ Testing routine. """
    # mode.test()
    print(model, test_data)




def main():
    """ Main function. """
    datasets = Datasets()

    if ARGS.model == 'cnn':
        model = CNNModel()
        model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 1)))

        # Print summary of model
        model.summary()
    elif ARGS.model == 'gan':
        model = GANModel()
        model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))

        # Print summary of model
        model.summary()

    if ARGS.evaluate:
        test(model, datasets)
    else:
        train(model, datasets)


# Make arguments global
ARGS = parse_args()

main()


# How to show images
# def main():
#     datasets = Datasets()
#     color, gray = datasets.load_data()
#     color1 = color[0]
#     gray1 = gray[0]
#     plt.imshow(color1)
#     plt.show()
#     plt.imshow(gray1, cmap="gray")
#     plt.show()

#     print(color.shape, gray.shape)
#     return
