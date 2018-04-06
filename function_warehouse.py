# This is place for the most common functions of this system

import numpy
from PIL import Image

def import_image(img_path):
    image_file = Image.open(img_path)
    image = image_file.convert('L')  # convert image to monochrome
    image = numpy.array(image)
    return image


def binarize_image(img_path, threshold):
    """Preprocess the image."""
    image_file = Image.open(img_path)
    image = image_file.convert('L')  # convert image to monochrome
    image = numpy.array(image)
    image = binarize_array(image, threshold)
    return image

def binarize_array(numpy_array, threshold):
    """Binarize a numpy array."""
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][j] > threshold:
                numpy_array[i][j] = 1
            else:
                numpy_array[i][j] = 0
    return numpy_array
