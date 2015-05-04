import numpy as np
from nolearn import BatchIterator
from params import *

class ScalingBatchIterator(BatchIterator):
	"""
	Scales images by subtracting mean and dividing by standard deviation.
	Note: Does not shuffle data.
	"""
	def __init__(self, mean, std, batch_size):
		super(ScalingBatchIterator, self).__init__(batch_size)

		self.mean = mean
		self.std = std

	def transform(self, Xb, yb):
		# Call super-class transform method. Currently this is just an identity function.
		Xb, yb = super(ScalingBatchIterator, self).transform(Xb, yb)

		# I encountered some bug in the past with changing the values of Xb directly, so I'm creating a copy here
		Xbb = np.zeros((Xb.shape[0], CHANNELS, PIXELS, PIXELS), dtype=np.float32)

		for i in range(Xb.shape[0]):
			Xbb[i, :, :, :] = (Xb[i, :, :, :] - self.mean) / self.std

		return Xbb, yb
