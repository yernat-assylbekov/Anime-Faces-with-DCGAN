"""
author: Yernat M. Assylbekov
email: yernat.assylbekov@gmail.com
date: 01/01/2020
"""


import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt


def read_images(data_dir):
    """
    reads images from data_dir into a numpy array of shape
    (number of images, height, width, number of channels (=3 for rgb)).
    pixels of each image are normalized (in the interval [0, 1]).
    """
    images = list()
    for file_name in glob.glob(data_dir):
        image = Image.open(file_name)
        image = np.asarray(image)
        images.append(image)

    return np.asarray(images) / 255.


def print_images(images):
    """
    prints first 9 images (in 3 columns and 3 rows) from images numpy array.
    """
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.axis('off')

    plt.show()
