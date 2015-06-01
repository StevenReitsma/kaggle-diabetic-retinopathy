from params import *
import numpy as np
import cv2
from scoop import shared, futures

import itertools
import time

from functools import partial


class Augmenter():
    def __init__(self):

        #Determine the center to rotate around
        self.center_shift = np.array((PIXELS, PIXELS)) / 2. - 0.5

    def augment(self, Xb):
        Xbb = np.zeros(Xb.shape, dtype=np.float32)

        # Random number 0-1 whether we flip or not
        random_flip = np.random.randint(2)

        # Translation shift
        shift_x = np.random.uniform(*AUGMENTATION_PARAMS['translation_range'])
        shift_y = np.random.uniform(*AUGMENTATION_PARAMS['translation_range'])

        # Rotation, zoom
        rotation = np.random.uniform(*AUGMENTATION_PARAMS['rotation_range'])
        log_zoom_range = [np.log(z) for z in AUGMENTATION_PARAMS['zoom_range']]
        zoom = np.exp(np.random.uniform(*log_zoom_range))

        # Color AUGMENTATION_PARAMS
        if COLOR_AUGMENTATION:
            random_hue = np.random.uniform(*AUGMENTATION_PARAMS['hue_range'])
            random_saturation = np.random.uniform(*AUGMENTATION_PARAMS['saturation_range'])
            random_value = np.random.uniform(*AUGMENTATION_PARAMS['value_range'])

        # Define affine matrix
        # TODO: Should be able to incorporate flips directly instead of through an extra call
        M = cv2.getRotationMatrix2D((self.center_shift[0], self.center_shift[1]), rotation, zoom)
        M[0, 2] += shift_x
        M[1, 2] += shift_y

        augment_partial = partial(augment_image,
                                    M=M,
                                    random_flip=random_flip,
                                    random_hue=random_hue,
                                    random_saturation=random_saturation,
                                    random_value=random_value)

        if CONCURRENT_AUGMENTATION:
            #FIXME: Currently not functional, can't mix and match multithreading
            augmented = futures.map(augment_partial, Xb)
            for index, im in enumerate(augmented):
                Xbb[index] = im
        else:
            for i in xrange(Xb.shape[0]):
                Xbb[i] = augment_partial(Xb[i])

        return Xbb



# Augments a single image, singled out for easier profiling
def augment_image(im, M=0, random_flip=0,
                    random_hue=0, random_saturation=0, random_value=0):

        im = cv2.warpAffine(im.transpose(1, 2, 0), M, (PIXELS, PIXELS))

        # im is now RGB 01c

        if random_flip == 1:
            im = cv2.flip(im, 0)


        if COLOR_AUGMENTATION:
            # Convert image to range 0-1.
            im = im / 255.

            # Convert to HSV
            im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)

            # Rescale hue from 0-360 to 0-1.
            im[:, :, 0] /= 360.

            # Mask value == 0
            black_indices = im[:, :, 2] == 0

            # Add random hue, saturation and value
            im[:, :, 0] = (im[:, :, 0] + random_hue) % 360
            im[:, :, 1] =  im[:, :, 1] + random_saturation
            im[:, :, 2] =  im[:, :, 2] + random_value

            im[black_indices, 2] = 0

            # Clip pixels from 0 to 1
            im = np.clip(im, 0, 1)

            if NETWORK_INPUT_TYPE == 'RGB':
                # Rescale hue from 0-1 to 0-360.
                im[:, :, 0] *= 360.

                # Convert back to RGB in 0-1 range.
                im = cv2.cvtColor(im, cv2.COLOR_HSV2RGB)

                # Convert back to 0-255 range
                im *= 255.

        # Back to c01
        return im.transpose(2, 0, 1)

        #if i % self.batch_size - 1 == 0:
            #scipy.misc.imsave('curimg.png', np.cast['int32'](Xbb[i]).transpose(1, 2, 0))
