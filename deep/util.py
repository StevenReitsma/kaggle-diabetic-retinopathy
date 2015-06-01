import numpy as np
import pandas as pd
from params import *
import os

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
