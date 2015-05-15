import numpy as np
import params

def float32(k):
	return np.cast['float32'](k)

def bin2array(_string):
	return np.fromstring(_string, dtype=np.uint8)
