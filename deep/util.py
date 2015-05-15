import numpy as np
import params
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
