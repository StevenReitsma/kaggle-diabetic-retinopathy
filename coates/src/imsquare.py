import numpy as np
import scipy
import scipy.misc

"""
Utility functions for squaring ndarrays (representing images)

Squaring here means making the width and height of the image equal, by
increasing one of either.

"""


# Square an image by stretching the shorter dimension
# Possible interpolation values: 'nearest', 'bilinear', 'bicubic'
# or 'cubic'
def square_stretch(image, interp= 'bilinear'):

    height = len(image)
    width = len(image[0])

    #Desired width and height length.
    desiredsize = max([height, width])

    return scipy.misc.imresize(image, (desiredsize, desiredsize), interp)



# Square an image by padding its sides
def square_pad(image, pad_value=255):

    height = len(image)
    width = len(image[0])

    #Desired width and height length.
    desired_size = max([height, width])


    if width < desired_size : # Pad to the left and right

        leftlength, rightlength = calc_pad_size(width, desired_size)


        lpad = np.empty([height, leftlength], dtype=int)
        rpad = np.empty([height, rightlength], dtype=int)

        rpad.fill(pad_value)
        lpad.fill(pad_value)

        # Horizontally stack the paddings around the image
        image = np.hstack((lpad, image, rpad))

    if height < desired_size :  # Pad to the top and bottom


        toplength, bottomlength = calc_pad_size(height, desired_size)

        tpad = np.empty([toplength, width], dtype=np.uint8)
        bpad = np.empty([bottomlength, width], dtype=np.uint8)

        tpad.fill(pad_value)
        bpad.fill(pad_value)

        # Vertically stack the paddings around the image
        image = np.vstack((tpad, image, bpad))

    return image

def get_square_function_by_name(name):
    if name == 'pad':
        return square_pad
    else:
        return square_stretch


# Returns the pad sizes for either side of the image
def calc_pad_size(current_length, desired_length):
    pad_length = desired_length - current_length

    l = pad_length // 2
    r = pad_length - l


    return l, r
