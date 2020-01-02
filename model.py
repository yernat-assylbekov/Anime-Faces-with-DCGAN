"""
author: Yernat M. Assylbekov
email: yernat.assylbekov@gmail.com
date: 01/01/2020
"""


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2DTranspose, Conv2D, LeakyReLU, Flatten, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam


def loss_generator(logit_fake):
    """
    loss function for generator.
    """
    return tf.math.negative(tf.math.reduce_mean(tf.math.log(logit_fake)))


def loss_discriminator(logit_real, logit_fake):
    """
    loss function for discriminator.
    """
    loss_real = tf.math.negative(tf.math.reduce_mean(tf.math.log(logit_real)))
    loss_fake = tf.math.negative(tf.math.reduce_mean(tf.math.log(1. - logit_fake)))
    return loss_real + loss_fake


def Generator(output_channels, noise_size):
    """
    model for generator.
    """

    # setup input
    X = Input(shape=noise_size)

    # project to 4x4
    Y = Dense(units=4*4*256)(X)
    Y = LeakyReLU()(Y)
    Y = BatchNormalization()(Y)
    Y = Reshape(target_shape=(4, 4, 256))(Y)

    # map to 8x8
    Y = Conv2DTranspose(filters=128, kernel_size=5, strides=2, padding='same', activation='relu')(Y)
    Y = BatchNormalization()(Y)

    # map to 16x16
    Y = Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same', activation='relu')(Y)
    Y = BatchNormalization()(Y)

    # map to 32x32
    Y = Conv2DTranspose(filters=32, kernel_size=5, strides=2, padding='same', activation='relu')(Y)
    Y = BatchNormalization()(Y)

    # map to 64x64
    Y = Conv2DTranspose(filters=output_channels, kernel_size=5, strides=2, padding='same', activation='sigmoid')(Y)

    model = Model(inputs=X, outputs=Y)

    return model

def Discriminator(input_channels):
    """
    model for discriminator.
    """

    # setup input
    X = Input(shape=(64, 64, input_channels))

    # map to 32x32
    Y = Conv2D(filters=32, kernel_size=5, strides=2, padding='same')(X) # , kernel_regularizer=l2(l=10.), bias_regularizer=l2(l=10.)
    Y = LeakyReLU()(Y)
    Y = BatchNormalization()(Y)
    Y = Dropout(0.4)(Y)

    # map to 16x16
    Y = Conv2D(filters=64, kernel_size=5, strides=2, padding='same')(Y)
    Y = LeakyReLU()(Y)
    Y = BatchNormalization()(Y)
    Y = Dropout(0.4)(Y)

    # map to 8x8
    Y = Conv2D(filters=128, kernel_size=5, strides=2, padding='same')(Y)
    Y = LeakyReLU()(Y)
    Y = BatchNormalization()(Y)
    Y = Dropout(0.4)(Y)

    # map to 4x4
    Y = Conv2D(filters=256, kernel_size=5, strides=2, padding='same')(Y)
    Y = LeakyReLU()(Y)
    Y = BatchNormalization()(Y)
    Y = Dropout(0.4)(Y)

    Y = Flatten()(Y)
    Y = Dense(units=1, activation='sigmoid')(Y)

    model = Model(inputs=X, outputs=Y)

    return model

def create_generator(channels, noise_size, learning_rate, beta_1):
    """
    creates generator, its optimizer (Adam) and checkpoint.
    """
    generator = Generator(output_channels=channels, noise_size=noise_size)
    generator_optimizer = Adam(lr=learning_rate, beta_1=beta_1)
    generator_checkpoint = tf.train.Checkpoint(optimizer=generator_optimizer, model=generator)

    return generator, generator_optimizer, generator_checkpoint

def create_discriminator(channels, learning_rate, beta_1):
    """
    creates discriminator, its optimizer (Adam) and checkpoint.
    """
    discriminator = Discriminator(input_channels=channels)
    discriminator_optimizer = Adam(lr=learning_rate, beta_1=beta_1)
    discriminator_checkpoint = tf.train.Checkpoint(optimizer=discriminator_optimizer, model=discriminator)

    return discriminator, discriminator_optimizer, discriminator_checkpoint
