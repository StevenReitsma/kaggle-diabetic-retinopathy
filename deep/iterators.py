import numpy as np
from nolearn import BatchIterator
from params import *
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
	"""
	Uses a producer-consumer model to prepare batches on the CPU while training on the GPU.

	If test = True, the test directory is taken to read the images and the transform method gets the
	keys sent as the second argument instead of the y_batch.
	"""

	def __init__(self, keys, batch_size, std, mean, y_all = None, test = False):
		self.batch_size = batch_size

		self.keys = keys
		self.y_all = y_all
		self.test = test
		self.mean = mean
		self.std = std

		if NETWORK_INPUT_TYPE == 'HSV':
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

			X_batch = np.zeros((cur_batch_size, CHANNELS, PIXELS, PIXELS), dtype=np.float32)
			y_batch = None

			if self.test:
				subdir = "test"
				y_batch = key_batch
			else:
				subdir = "train"
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
	def __init__(self, keys, batch_size, std, mean, y_all = None):
		super(AugmentingParallelBatchIterator, self).__init__(keys, batch_size, std, mean, y_all)

		# Set center point
		self.center_shift = np.array((PIXELS, PIXELS)) / 2. - 0.5

	def transform(self, Xb, yb):
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

		# For every image, perform the actual warp, per channel
		for i in xrange(Xb.shape[0]):
			im = cv2.warpAffine(Xb[i].transpose(1, 2, 0), M, (PIXELS, PIXELS))

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

				# Add random hue, saturation and value
				im[:, :, 0] = im[:, :, 0] + random_hue
				im[:, :, 1] = im[:, :, 1] + random_saturation
				im[:, :, 2] = im[:, :, 2] + random_value

				# Clip pixels from 0 to 1
				im = np.clip(im, 0, 1)

				if NETWORK_INPUT_TYPE == 'RGB':
					# Rescale hue from 0-1 to 0-360.
					im[:, :, 0] *= 360.

					# Convert back to RGB in 0-1 range.
					im = cv2.cvtColor(im, cv2.COLOR_HSV2RGB)

					# Convert back to 0-255 range
					im *= 255.

					print im.max(), im.min(), im.mean()

			# Back to c01
			Xbb[i] = im.transpose(2, 0, 1)

		# Do normalization in super-method
		Xbb, yb = super(AugmentingParallelBatchIterator, self).transform(Xbb, yb)

		return Xbb, yb

class TTABatchIterator(ParallelBatchIterator):
	def __init__(self, keys, batch_size, std, mean):
		super(TTABatchIterator, self).__init__(keys, batch_size, std, mean, test = True)

		# Set center point
		self.center_shift = np.array((PIXELS, PIXELS)) / 2. - 0.5
		self.i = 0

	def transform(self, Xb, yb):
		print "Batch %i/%i" % (self.i, self.X.shape[0]/self.batch_size)
		self.i += 1

		# Create some augmented batches
		rotations = [0, 45, 90, 135, 180, 225, 270, 315]
		flips = [True, False]

		self.ttas = len(rotations) * len(flips)

		Xbb_list = []

		for r in rotations:
			for f in flips:
				Xbb_new = np.zeros(Xb.shape, dtype=np.float32)

				M = cv2.getRotationMatrix2D((self.center_shift[0], self.center_shift[1]), r, 1)

				for i in range(Xb.shape[0]):
					im = cv2.warpAffine(Xb[i].transpose(1, 2, 0), M, (PIXELS, PIXELS))
					if f:
						Xbb_new[i] = cv2.flip(im, 0).transpose(2, 0, 1)
					else:
						Xbb_new[i] = im.transpose(2, 0, 1)

				# Normalize
				Xbb_new, _ = super(TTABatchIterator, self).transform(Xbb_new, None)

				# Extend if batch size too small
				if Xbb_new.shape[0] < self.batch_size:
					Xbb_new = np.vstack([Xbb_new, np.zeros((self.batch_size - Xbb_new.shape[0], Xbb_new.shape[1], Xbb_new.shape[2], Xbb_new.shape[3]), dtype=np.float32)])

				Xbb_list.append(Xbb_new)

		# yb are the keys of this batch, in order.
		return np.vstack(Xbb_list), yb
