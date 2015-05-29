# -*- coding: utf-8 -*-
import numpy as np
from scipy import misc
# Misc image functions

def flatten_image(image):
    return np.ravel(image)    
    
    
def resize_image(image, size=32, interp='bilinear'):
    return ( misc.imresize(image, (size, size), interp))

def sum_images(sequence_of_images):
    return sum(sequence_of_images)
    
def normalize(image, mean_image, std_image):
    return (image - mean_image) / std_image
    
def image_horizontal(image):
    if len(image) > len(image[0]):
        image = np.rot90(image)
        
    return image