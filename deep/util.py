import numpy as np
import pandas as pd
from params import *
import os
import cv2

def float32(k):
	return np.cast['float32'](k)

def bin2array(_string):
	return np.fromstring(_string, dtype=np.uint8)

def load_labels():
	# Read CSV file with labels
	return pd.DataFrame.from_csv(
		os.path.join(IMAGE_SOURCE, "..", "trainLabels.csv"))

def load_sample_submission():
	return pd.DataFrame.from_csv(
		os.path.join(IMAGE_SOURCE, "..", "sampleSubmission.csv"))

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
        from http://goo.gl/DZNhk
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def hsv_augment(im, hue, saturation, value):
	"""
	Augments an image with additive hue, saturation and value.

	`im` should be 01c RGB in range 0-255.
	`hue`, `saturation` and `value` should be scalars between -1 and 1.

	Return value: if NETWORK_INPUT_TYPE is RGB, a 01c RGB image. If NETWORK_INPUT_TYPE is HSV, a 01c HSV image.
	"""

	# Convert image to range 0-1.
	im = im / 255.

	# Convert to HSV
	im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)

	# Rescale hue from 0-360 to 0-1.
	im[:, :, 0] /= 360.

    # Mask value == 0
	black_indices = im[:, :, 2] == 0

	# Add random hue, saturation and value
	im[:, :, 0] = (im[:, :, 0] + hue) % 1
	im[:, :, 1] = im[:, :, 1] + saturation
	im[:, :, 2] = im[:, :, 2] + value

	# Pixels that were black stay black
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

	return im
