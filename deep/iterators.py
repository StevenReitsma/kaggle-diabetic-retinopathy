import numpy as np
from nolearn import BatchIterator
import matplotlib.pyplot as plt
from params import *
import skimage
from skimage import transform
import scipy
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
	def __init__(self, keys, y_all, batch_size, std, mean):
		self.batch_size = batch_size

		self.keys = keys
		self.mean = mean
		self.std = std
		self.y_all = y_all

	def __call__(self, X, y=None):
		self.X = X
		self.y = y
		return self

	def gen(self):
		n_samples = self.X.shape[0]
		bs = self.batch_size

		for i in xrange((n_samples + bs - 1) // bs):
			t = time.time()
			start_index = i * bs
			end_index = (i+1) * bs

			indices = self.X[start_index:end_index]
			key_batch = self.keys[indices]

			cur_batch_size = len(indices)

			X_batch = np.zeros((cur_batch_size, CHANNELS, PIXELS, PIXELS), dtype=np.float32)
			y_batch = None

			#subdir = "train" if self.y is not None else "test"
			subdir = "train"

			if self.y is not None:
				y_batch = self.y_all.loc[key_batch]['level']
				y_batch = y_batch[:, np.newaxis].astype(np.float32)

			# Read all images in the batch
			for i, key in enumerate(key_batch):
				X_batch[i] = scipy.misc.imread(IMAGE_SOURCE + "/" + subdir + "/" + key + ".jpeg").transpose(2, 0, 1)

			# Transform the batch (augmentation, normalization, etc.)
			X_batch, y_batch = self.transform(X_batch, y_batch)

			#print "Produce time: %.2f ms" % ((time.time() - t)*1000)

			yield X_batch, y_batch

	def __iter__(self):
		import Queue
		queue = Queue.Queue(maxsize=16)
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
		# Normalize
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
	def __init__(self, keys, y_all, batch_size, std, mean):
		super(AugmentingParallelBatchIterator, self).__init__(keys, y_all, batch_size, std, mean)

		# Set center point
		self.center_shift = np.array((PIXELS, PIXELS)) / 2. - 0.5
		self.tform_center = transform.SimilarityTransform(translation=-self.center_shift)
		self.tform_uncenter = transform.SimilarityTransform(translation=self.center_shift)

		# Identities
		self.tform_identity = skimage.transform.AffineTransform()
		self.tform_ds = skimage.transform.AffineTransform()

	def transform(self, Xb, yb):
		Xbb = np.zeros((Xb.shape[0], Xb.shape[1], Xb.shape[2], Xb.shape[3]), dtype=np.float32)

		# Random number 0-1 whether we flip or not
		random_flip = np.random.randint(2)

		# Translation shift
		shift_x = np.random.uniform(*AUGMENTATION_PARAMS['translation_range'])
		shift_y = np.random.uniform(*AUGMENTATION_PARAMS['translation_range'])

		# Rotation, zoom
		rotation = np.random.uniform(*AUGMENTATION_PARAMS['rotation_range'])
		log_zoom_range = [np.log(z) for z in AUGMENTATION_PARAMS['zoom_range']]
		zoom = np.exp(np.random.uniform(*log_zoom_range))
		
		# Define affine matrix
		M = cv2.getRotationMatrix2D((self.center_shift[0], self.center_shift[1]), rotation, zoom)
		M[0, 2] += shift_x
		M[1, 2] += shift_y

		# For every image, perform the actual warp, per channel
		for i in range(Xb.shape[0]):
			Xbb[i] = cv2.warpAffine(Xb[i].transpose(1, 2, 0), M, (PIXELS, PIXELS)).transpose(2, 0, 1)

			if random_flip == 1:
				Xbb[i] = cv2.flip(Xbb[i].transpose(1, 2, 0), 0).transpose(2, 0, 1)

		# Do normalization in super-method
		Xbb, yb = super(AugmentingParallelBatchIterator, self).transform(Xbb, yb)

		return Xbb, yb
