"""
author: Yernat M. Assylbekov
email: yernat.assylbekov@gmail.com
date: 01/01/2020
"""


import numpy as np
import tensorflow as tf
from model import Generator, Discriminator, loss_generator, loss_discriminator, create_generator, create_discriminator
from utils import read_images, print_images
from IPython import display
import os
import matplotlib.pyplot as plt


def generate_print_save(model, input, epoch):
    """
    generates, prints and saves 9 images for a given input noise of shape [9, noise_size],
    where noise_size is globally defined.
    """
    prediction = model(input, training=False)
    fig = plt.figure(figsize=(10, 10))

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(prediction[i])
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch+1))
    plt.show()


def train_DCGAN(train_set, batch_size, epochs):
    """
    trains the globally defined generator and discriminator.
    """
    # train set size
    train_size = train_set.shape[0]

    # the size of noise vector is given globally
    global noise_size

    # number of mini batches
    m = train_size // batch_size

    # partition the training set to mini batches
    train_batches = np.split(train_set, [k * batch_size for k in range(1, m)])

    # generator and discriminator, their optimizers and checkpoints are given globally
    global generator, generator_optimizer, generator_checkpoint
    global discriminator, discriminator_optimizer, discriminator_checkpoint

    # prefixes for the checkpoints of the generator and discriminator
    generator_prefix = os.path.join('./generator', 'ckpt')
    discriminator_prefix = os.path.join('./discriminator', 'ckpt')

    # lists to record costs of the generator and discriminator at every 10 epochs
    generator_costs = list()
    discriminator_costs = list()

    for epoch in range(epochs):

        # initiate costs of the generator and discriminator at the current epoch to zeros
        generator_cost = 0
        discriminator_cost = 0

        for batch in train_batches:
            # random noise for the current mini batch
            noise = tf.random.normal([batch_size, noise_size])

            # watch trainable variables for the loss functions of the generator and discriminator
            with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
                fake_images = generator(noise, training=True) # the generator generates fake images

                # the discriminator computes logits (probabilites) of images for being real
                logit_fake = discriminator(fake_images, training=True)
                logit_real = discriminator(batch, training=True)

                # loss functions for the generator and discriminator
                generator_loss = loss_generator(logit_fake)
                discriminator_loss = loss_discriminator(logit_real, logit_fake)

            # compute gradients and perform one step gradient descend
            generator_Grads = generator_tape.gradient(generator_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(generator_Grads, generator.trainable_variables))
            discriminator_Grads = discriminator_tape.gradient(discriminator_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(discriminator_Grads, discriminator.trainable_variables))

            # record costs of the generator and discriminator at every 10 epochs
            if (epoch + 1) % 10 == 0:
                fake_images = generator(noise, training=False)
                logit_fake = discriminator(fake_images, training=False)
                logit_real = discriminator(batch, training=False)
                generator_cost += loss_generator(logit_fake).numpy() / m
                discriminator_cost += loss_discriminator(logit_real, logit_fake).numpy() / m

        # save checkpoints at every 10 epochs
        # also print and save 9 randomly generated images at every 10 epochs
        if (epoch + 1) % 10 == 0:
            generator_checkpoint.save(file_prefix=generator_prefix)
            discriminator_checkpoint.save(file_prefix=discriminator_prefix)

            generator_costs.append(generator_cost)
            discriminator_costs.append(discriminator_cost)

            display.clear_output(wait=True)
            print('Epoch: {}'.format(epoch+1))
            print('Generator loss: {}'.format(generator_loss))
            print('Discriminator loss: {}'.format(discriminator_loss))
            noise = tf.random.normal([9, noise_size])
            generate_print_save(generator, noise, epoch)

    # plot the learning curves of the generator and discriminator
    plt.plot(np.squeeze(generator_costs))
    plt.plot(np.squeeze(discriminator_costs))
    plt.show()

# download images
train_set = read_images(data_dir='data/*.png')

# let us look at few images from the dataset
print_images(images=train_set)

# setup number of channels, noise size, learning rate and beta_1
channels = 3
noise_size = 100
learning_rate = 0.0001
beta_1 = 0.5

# create an instance of generator and discriminator, their optimizers and checkpoints
generator, generator_optimizer, generator_checkpoint = create_generator(channels, noise_size, learning_rate, beta_1)
discriminator, discriminator_optimizer, discriminator_checkpoint = create_discriminator(channels, learning_rate, beta_1)

# train generator and discriminator
train_DCGAN(train_set, batch_size=100, epochs=500)
