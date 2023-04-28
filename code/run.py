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

from tensorboard_utils import \
        ImageLabelingLogger, ConfusionMatrixLogger, CustomModelSaver

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
        choices=['cnn', 'cnn-pre', 'gan'],
        help='''Which model to use''')
    parser.add_argument(
        '--data',
        default='TODO',
        help='Location where the dataset is stored.')
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights.''')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')
    return parser.parse_args()


def train(model, datasets, checkpoint_path, init_epoch):
    """ Training routine. """
    callback_list = [
        # tf.keras.callbacks.TensorBoard(
        #     log_dir=logs_path,
        #     update_freq='batch',
        #     profile_batch=0),
        # ImageLabelingLogger(logs_path, datasets),
        CustomModelSaver(checkpoint_path, ARGS.model, hp.max_num_weights)
    ]

    print("Fit model on training data")
    history = model.fit(
        datasets.train_gray,
        datasets.train_color,
        batch_size=64,
        epochs=hp.num_epochs,
        callbacks=callback_list,
        initial_epoch=init_epoch,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(datasets.validation_gray, datasets.validation_color),
    )

    print(history.history)


def test(model, datasets):
    """ Testing routine. """
    model.evaluate(
        x=datasets.test_gray,
        y=datasets.test_color,
        verbose=1,
    )
    #print(model, test_data)


def main():
    """ Main function. """

    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    # If loading from a checkpoint, the loaded checkpoint's directory
    # will be used for future checkpoints
    if ARGS.load_checkpoint is not None:
        ARGS.load_checkpoint = os.path.abspath(ARGS.load_checkpoint)

        # Get timestamp and epoch from filename
        regex = r"(?:.+)(?:\.e)(\d+)(?:.+)(?:.h5)"
        init_epoch = int(re.match(regex, ARGS.load_checkpoint).group(1)) + 1
        timestamp = os.path.basename(os.path.dirname(ARGS.load_checkpoint))

    datasets = Datasets()
    print("Completed loading datasets")

    if ARGS.model == 'cnn':
        model = CNNModel()
        model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 1)))
        checkpoint_path = "checkpoints" + os.sep + \
            "your_model" + os.sep + timestamp + os.sep

        # Print summary of model
        model.summary()
    elif ARGS.model == 'cnn-pre':
        # pre-trained CNN
        model = CNNModel()
        model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 1)))

        # Print summary of model
        model.summary()
    elif ARGS.model == 'gan':
        model = GANModel()
        model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))

        # Print summary of model
        model.summary()

    # Load checkpoints
    if ARGS.load_checkpoint is not None:
        if ARGS.model == 'cnn':
            model.load_weights(ARGS.load_checkpoint, by_name=False)
        # TODO: Add elif cases for other models
    
    # Make checkpoint directory if needed
    if not ARGS.evaluate and not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # TODO add options for non-CNN models
    model.compile(
        optimizer=model.optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=["mean_squared_error"])

    if ARGS.evaluate:
        test(model, datasets)
    else:
        train(model, datasets, checkpoint_path, init_epoch)


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
