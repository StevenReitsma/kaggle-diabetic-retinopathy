import numpy as np
from nolearn.lasagne import BatchIterator
from params import *
import scipy
from augment import Augmenter
import cv2

import util
import time

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

		# Normalize
		Xbb = (Xb - self.mean) / self.std

		return Xbb, yb

class ParallelBatchIterator(object):
	"""
	Uses a producer-consumer model to prepare batches on the CPU while training on the GPU.

	If test = True, the test directory is taken to read the images and the transform method gets the
	keys sent as the second argument instead of the y_batch.
	"""

	def __init__(self, keys, batch_size, std, mean, coates_features = None, y_all = None, test = False, cv = False):
		self.batch_size = batch_size

		self.keys = keys
		self.y_all = y_all
		self.test = test
		self.mean = mean
		self.std = std
		self.cv = cv
		self.coates_features = coates_features

		if params.NETWORK_INPUT_TYPE == 'HSV':
			self.mean = self.mean / 255.
			self.std = self.std / 255.
			self.mean = cv2.cvtColor(self.mean.transpose(1, 2, 0), cv2.COLOR_RGB2HSV).transpose(2, 0, 1)
			self.std = cv2.cvtColor(self.std.transpose(1, 2, 0), cv2.COLOR_RGB2HSV).transpose(2, 0, 1)

	def __call__(self, X, y=None):
		self.X = X
		self.y = y
		return self

	def gen(self):
		n_samples = self.X.shape[0]
		bs = self.batch_size

		for i in xrange((n_samples + bs - 1) // bs):
			#t = time.time()
			start_index = i * bs
			end_index = (i+1) * bs

			indices = self.X[start_index:end_index]
			key_batch = self.keys[indices]

			cur_batch_size = len(indices)

			X_batch = np.zeros((cur_batch_size, params.CHANNELS, params.PIXELS, params.PIXELS), dtype=np.float32)
			y_batch = None

			if self.test:
				subdir = "test"
				y_batch = key_batch
			else:
				subdir = "train"
				y_batch = self.y_all.loc[key_batch]['level']
				y_batch = y_batch[:, np.newaxis].astype(np.float32)

			if self.cv:
				subdir = "train"

			# Read all images in the batch
			for i, key in enumerate(key_batch):
				X_batch[i] = scipy.misc.imread(params.IMAGE_SOURCE + "/" + subdir + "/" + key + ".jpeg").transpose(2, 0, 1)

			# Transform the batch (augmentation, normalization, etc.)
			X_batch, y_batch = self.transform(X_batch, y_batch)

			#print "Produce time: %.2f ms" % ((time.time() - t)*1000)

			if self.coates_features is not None:
				# Get Coates
				coates_batch = np.array([self.coates_features[k] for k in key_batch])
				coates_batch = coates_batch.reshape(cur_batch_size, 1, 1, -1)

				yield {'input': X_batch, 'coates': coates_batch}, y_batch
			else:
				yield X_batch, y_batch

	def __iter__(self):
		import Queue
		queue = Queue.Queue(maxsize=8)
		sentinel = object()  # guaranteed unique reference

		# Define producer (putting items into queue)
		def producer():
			for item in self.gen():
				queue.put(item)
				#print ">>>>> P:\t%i" % (queue.qsize())
			queue.put(sentinel)

		# Start producer (in a background thread)
		import threading
		thread = threading.Thread(target=producer)
		thread.daemon = True
		thread.start()

		# Run as consumer (read items from queue, in current thread)
		item = queue.get()
		while item is not sentinel:
			yield item
			queue.task_done()
			item = queue.get()
			#print "C:\t%i" % (queue.qsize())

	def transform(self, Xb, yb):
		Xbb = (Xb - self.mean) / self.std

		return Xbb, yb

class RedisIterator():
	def __init__(self, redis, keys):
		self.r = redis
		self.keys = keys

	def __iter__(self):
		for key in self.keys:
			_string = self.r.get(key)
			_dat = util.bin2array(_string)
			yield _dat

class AugmentingParallelBatchIterator(ParallelBatchIterator):
	"""
	Randomly changes images in the batch. Behaviour can be defined in params.py.
	"""
	def __init__(self, keys, batch_size, std, mean, coates_features = None, y_all = None):
		super(AugmentingParallelBatchIterator, self).__init__(keys, batch_size, std, mean, coates_features, y_all)

		# Initialize augmenter
		self.augmenter = Augmenter()

	def transform(self, Xb, yb):
		Xbb = self.augmenter.augment(Xb)

		# Do normalization in super-method
		Xbb, yb = super(AugmentingParallelBatchIterator, self).transform(Xbb, yb)

		return Xbb, yb

class TTABatchIterator(ParallelBatchIterator):
	def __init__(self, keys, batch_size, std, mean, coates_features = None, cv = False):
		super(TTABatchIterator, self).__init__(keys, batch_size, std, mean, coates_features = coates_features,  test = True, cv = cv)

		# Set center point
		self.center_shift = np.array((PIXELS, PIXELS)) / 2. - 0.5
		self.i = 0

		self.rotations = [0, 45, 90, 135, 180, 225, 270, 315]
		self.flips = [True, False]
		self.hue = [0]
		self.saturation = [0]

		self.ttas = len(self.rotations) * len(self.flips) * len(self.hue) * len(self.saturation)

	def transform(self, Xb, yb):
		print "Batch %i/%i" % (self.i, self.X.shape[0]/self.batch_size)
		self.i += 1

		Xbb_list = []

		for h in self.hue:
			for s in self.saturation:
				for r in self.rotations:
					for f in self.flips:
						Xbb_new = np.zeros(Xb.shape, dtype=np.float32)

						M = cv2.getRotationMatrix2D((self.center_shift[0], self.center_shift[1]), r, 1)

						for i in range(Xb.shape[0]):
							im = cv2.warpAffine(Xb[i].transpose(1, 2, 0), M, (params.PIXELS, params.PIXELS))

							if f:
								im = cv2.flip(im, 0)

							Xbb_new[i] = util.hsv_augment(im, h, s, 0).transpose(2, 0, 1)

						# Normalize
						Xbb_new, _ = super(TTABatchIterator, self).transform(Xbb_new, None)

						# Extend if batch size too small
						if Xbb_new.shape[0] < self.batch_size:
							Xbb_new = np.vstack([Xbb_new, np.zeros((self.batch_size - Xbb_new.shape[0], Xbb_new.shape[1], Xbb_new.shape[2], Xbb_new.shape[3]), dtype=np.float32)])

						Xbb_list.append(Xbb_new)

		# yb are the keys of this batch, in order.
		return np.vstack(Xbb_list), yb
