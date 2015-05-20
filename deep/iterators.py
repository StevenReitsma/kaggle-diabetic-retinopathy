import numpy as np
from nolearn import BatchIterator
from params import *
import skimage
from skimage import transform
import scipy

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
		center_shift = np.array((PIXELS, PIXELS)) / 2. - 0.5
		self.tform_center = transform.SimilarityTransform(translation=-center_shift)
		self.tform_uncenter = transform.SimilarityTransform(translation=center_shift)

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

		# Rotation, shear, zoom shift
		translation = (shift_x, shift_y)
		rotation = np.random.uniform(*AUGMENTATION_PARAMS['rotation_range'])
		shear = np.random.uniform(*AUGMENTATION_PARAMS['shear_range'])
		log_zoom_range = [np.log(z) for z in AUGMENTATION_PARAMS['zoom_range']]
		zoom = np.exp(np.random.uniform(*log_zoom_range))

		# Whether to do flipping
		# Flipping is equivalent to shearing 180 degrees and rotating 180 degrees
		if AUGMENTATION_PARAMS['do_flip'] and random_flip > 0:
			shear += 180
			rotation += 180

		# Create augmentation transformer
		tform_augment = transform.AffineTransform(scale=(1/zoom, 1/zoom), rotation=np.deg2rad(rotation), shear=np.deg2rad(shear), translation=translation)
		tform_augment = self.tform_center + tform_augment + self.tform_uncenter

		def fast_warp(img, tf, output_shape=(PIXELS, PIXELS), mode='nearest'):
			return skimage.transform._warps_cy._warp_fast(img, tf.params, output_shape=output_shape, mode=mode)

		# For every image, perform the actual warp, per channel
		for i in range(Xb.shape[0]):
			for c in range(Xb.shape[1]):
				Xbb[i, c, :, :] = fast_warp(Xb[i][c], self.tform_ds + tform_augment + self.tform_identity).astype('float32')

		# Do normalization in super-method
		Xbb, yb = super(AugmentingParallelBatchIterator, self).transform(Xbb, yb)

		return Xbb, yb
