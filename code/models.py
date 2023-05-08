"""
Homework 5 - CNNs
CS1430 - Computer Vision
Brown University
"""

import tensorflow as tf
from keras.layers import \
    Conv2D, MaxPool2D, Flatten, Dense, UpSampling2D, BatchNormalization, LeakyReLU
import numpy as np
import hyperparameters as hp


class CNNModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(CNNModel, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.architecture = [
            Conv2D(filters=64, kernel_size=(3, 3),
                   activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(filters=64, kernel_size=(3, 3),
                   activation='relu', padding='same'),
            BatchNormalization(),
            MaxPool2D(pool_size=(2, 2)),
            Conv2D(filters=128, kernel_size=(3, 3),
                   activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(filters=128, kernel_size=(3, 3),
                   activation='relu', padding='same'),
            BatchNormalization(),
            MaxPool2D(pool_size=(2, 2)),
            Conv2D(filters=256, kernel_size=(3, 3),
                   activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(filters=256, kernel_size=(3, 3),
                   activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(filters=256, kernel_size=(3, 3),
                   activation='relu', padding='same'),
            BatchNormalization(),
            MaxPool2D(pool_size=(2, 2)),
            Conv2D(filters=512, kernel_size=(3, 3),
                   activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(filters=512, kernel_size=(3, 3),
                   activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(filters=512, kernel_size=(3, 3),
                   activation='relu', padding='same'),
            BatchNormalization(),
            MaxPool2D(pool_size=(2, 2)),

            UpSampling2D(size=(2, 2)),
            Conv2D(filters=256, kernel_size=(3, 3),
                   activation='relu', padding='same'),
            BatchNormalization(),
            UpSampling2D(size=(2, 2)),
            Conv2D(filters=128, kernel_size=(3, 3),
                   activation='relu', padding='same'),
            BatchNormalization(),
            UpSampling2D(size=(2, 2)),
            Conv2D(filters=64, kernel_size=(3, 3),
                   activation='relu', padding='same'),
            BatchNormalization(),
            UpSampling2D(size=(2, 2)),
            Conv2D(filters=32, kernel_size=(3, 3),
                   activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(filters=2, kernel_size=(1, 1),
                   activation='relu', padding='same'),

            Flatten(),
            Dense(units=512, activation="relu"),
            Dense(units=112 * 112 * 2, activation="relu"),
            tf.keras.layers.Reshape((112, 112, 2))
        ]

    def call(self, x):
        """ Passes input image through the network. """
        for layer in self.architecture:
            x = layer(x)
        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """
        return tf.keras.losses.MeanSquaredError(labels, predictions)


class RESCNNModel(tf.keras.Model):
    """ CNN that uses ResNet """

    def __init__(self):
        super(RESCNNModel, self).__init__()
        self.optimizer = tf.keras.optimizers.SGD()
        self.RES = tf.keras.applications.resnet50.ResNet50(
            include_top=False,
            input_shape=(hp.img_size, hp.img_size, 3))

        for layer in self.RES.layers:
            layer.trainable = False

        self.head = [
            # input size: (4, 4, 2048)
            Conv2D(256, 3, padding="same"),
            BatchNormalization(),
            LeakyReLU(),
            UpSampling2D(size=(7, 7)),
            # (28, 28, 256)
            Conv2D(128, 3, padding="same"),
            BatchNormalization(),
            LeakyReLU(),
            UpSampling2D(size=(2, 2)),
            # (56, 56, 128)
            Conv2D(64, 3, padding="same"),
            BatchNormalization(),
            LeakyReLU(),
            UpSampling2D(size=(2, 2)),
            # (112, 112, 64)
            Conv2D(32, 3, padding="same"),
            BatchNormalization(),
            LeakyReLU(),
            # (112, 112, 32)
            Conv2D(2, 3, padding="same"),
            # (112, 112, 2)
        ]

        self.model = tf.keras.Sequential(name="RES")
        self.model.add(self.RES)
        self.head = tf.keras.Sequential(self.head, name="vgg_head")

    def call(self, x):
        """ Passes input image through the network. """
        x = tf.concat((x,)*3, axis=-1)
        x = self.model(x)
        x = self.head(x)
        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """
        return tf.keras.losses.MeanAbsoluteError(labels, predictions)


class GeneratorModel(tf.keras.Model):
    """ CNN that uses ResNet """

    def __init__(self):
        super(GeneratorModel, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(1e-2)
        self.RES = tf.keras.applications.resnet50.ResNet50(
            include_top=False,
            input_shape=(hp.img_size, hp.img_size, 3))

        for layer in self.RES.layers:
            layer.trainable = False

        self.head = [
            # input size: (4, 4, 2048)
            Conv2D(256, 3, padding="same"),
            BatchNormalization(),
            LeakyReLU(),
            UpSampling2D(size=(7, 7)),
            # (28, 28, 256)
            Conv2D(128, 3, padding="same"),
            BatchNormalization(),
            LeakyReLU(),
            UpSampling2D(size=(2, 2)),
            # (56, 56, 128)
            Conv2D(64, 3, padding="same"),
            BatchNormalization(),
            LeakyReLU(),
            UpSampling2D(size=(2, 2)),
            # (112, 112, 64)
            Conv2D(32, 3, padding="same"),
            BatchNormalization(),
            LeakyReLU(),
            # (112, 112, 32)
            Conv2D(2, 3, padding="same"),
            # (112, 112, 2)
        ]

        self.model = tf.keras.Sequential(name="RES")
        self.model.add(self.RES)
        self.head = tf.keras.Sequential(self.head, name="vgg_head")

    def call(self, x):
        """ Passes input image through the network. """
        x = tf.concat((x,)*3, axis=-1)
        x = self.model(x)
        x = self.head(x)
        return x


class GANModel():
    """ Your own neural network model. """

    def __init__(self):
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy()
        self.generator_opt = tf.keras.optimizers.Adam(1e-2)
        self.discriminator_opt = tf.keras.optimizers.Adam(1e-2)
        self.generator = GeneratorModel()
        self.discriminator = self.make_discriminator_model()

        self.generator.compile(
            optimizer=self.generator_opt,
            loss=tf.keras.losses.MeanAbsoluteError(),
            metrics=["mean_squared_error"])

    def make_resnet_model(self):
        RES = tf.keras.applications.resnet50.ResNet50(
            include_top=False,
            input_shape=(hp.img_size, hp.img_size, 3))

        for layer in RES.layers:
            layer.trainable = False

        self.model = tf.keras.Sequential(name="RES")

    def call(self, x):
        """ Passes input image through the network. """
        x = tf.concat((x,)*3, axis=-1)
        x = self.model(x)
        x = self.head(x)
        return x

    def make_discriminator_model(self):
        model = [
            Conv2D(32, 3, padding="same"),
            BatchNormalization(),
            LeakyReLU(),
            MaxPool2D(pool_size=(2, 2)),
            Conv2D(64, 3, padding="same"),
            BatchNormalization(),
            LeakyReLU(),
            Conv2D(128, 3, padding="same"),
            BatchNormalization(),
            LeakyReLU(),
            Conv2D(256, 3, padding="same"),
            BatchNormalization(),
            LeakyReLU(),
            MaxPool2D(pool_size=(2, 2)),
            Flatten(),
            Dense(256, activation="leaky_relu"),
            Dense(64, activation="leaky_relu"),
            Dense(1, activation="sigmoid")
        ]
        model = tf.keras.Sequential(model)

        return model

    def generator_loss(self, fake_out, real_out, prob_fake):
        cross_entropy_loss = self.cross_entropy(
            tf.ones_like(prob_fake), prob_fake)
        # Change to mean
        l2 = tf.keras.losses.MeanSquaredError()
        l2 = l2(fake_out, real_out)
        return (0.1 * l2) + cross_entropy_loss

    def discriminator_loss(self, real_out, fake_out):
        real_loss = self.cross_entropy(tf.ones_like(real_out), real_out)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_out), fake_out)
        total_loss = real_loss + fake_loss

        return total_loss

    def train_generator_step(self, L_batch, ab_batch):
        with tf.GradientTape() as g_tape:
            fake_ab = self.generator(L_batch, training=True)

            real_LAB = np.concatenate((L_batch, ab_batch), axis=-1)
            fake_LAB = np.concatenate((L_batch, fake_ab), axis=-1)

            prob_fake = self.discriminator(fake_LAB, training=True)

            g_loss = self.generator_loss(fake_ab, ab_batch, prob_fake)

        # Compute gradients
        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)

        # Optimize generator
        self.generator_opt.apply_gradients(
            zip(g_grads, self.generator.trainable_variables))

        return g_loss

    def train_discriminator_step(self, L_batch, ab_batch):
        with tf.GradientTape() as d_tape:
            fake_ab = self.generator(L_batch, training=True)

            real_LAB = np.concatenate((L_batch, ab_batch), axis=-1)
            fake_LAB = np.concatenate((L_batch, fake_ab), axis=-1)

            prob_real = self.discriminator(real_LAB, training=True)
            prob_fake = self.discriminator(fake_LAB, training=True)

            d_loss = self.discriminator_loss(prob_real, prob_fake)

        # Compute gradients
        d_grads = d_tape.gradient(
            d_loss, self.discriminator.trainable_variables)

        # Optimize discriminator
        self.discriminator_opt.apply_gradients(
            zip(d_grads, self.discriminator.trainable_variables))

        return d_loss

    # def train_step(self, L_batch, ab_batch):
    #     with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
    #         fake_ab = self.generator(L_batch, training=True)

    #         real_LAB = np.concatenate((L_batch, ab_batch), axis=-1)
    #         fake_LAB = np.concatenate((L_batch, fake_ab), axis=-1)
    #         # Display
    #         # L = L_batch[0]
    #         # fake_ab_img = fake_ab[0]
    #         # real_ab_img = ab_batch[0]

    #         # base_LAB = np.concatenate((L,)*3, axis=-1)
    #         # base_LAB[:, :, 1:] = real_ab_img
    #         # real_RGB = lab2rgb(base_LAB)

    #         # base_LAB[:, :, 1:] = fake_ab_img
    #         # fake_RGB = lab2rgb(base_LAB)

    #         # plt.imshow(L, cmap="gray")
    #         # plt.show()
    #         # plt.imshow(real_RGB)
    #         # plt.show()
    #         # plt.imshow(fake_RGB)
    #         # plt.show()
    #         #

    #         # Train the generator and discriminator seperately
    #         prob_real = self.discriminator(real_LAB, training=True)
    #         prob_fake = self.discriminator(fake_LAB, training=True)

    #         g_loss = self.generator_loss(fake_ab, ab_batch, prob_fake)
    #         d_loss = self.discriminator_loss(prob_real, prob_fake)

    #     # Compute gradients
    #     g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
    #     d_grads = d_tape.gradient(
    #         d_loss, self.discriminator.trainable_variables)

    #     # Optimize
    #     self.generator_opt.apply_gradients(
    #         zip(g_grads, self.generator.trainable_variables))
    #     self.discriminator_opt.apply_gradients(
    #         zip(d_grads, self.discriminator.trainable_variables))

    #     return g_loss, d_loss
