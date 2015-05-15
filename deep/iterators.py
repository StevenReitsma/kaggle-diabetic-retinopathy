import numpy as np
from nolearn import BatchIterator
from params import *
import skimage
from skimage import transform

import util

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
	def __init__(self, keys, r, batch_size, std, mean):
		self.batch_size = batch_size

		self.keys = keys
		self.mean = mean
		self.std = std
		self.r = r

		# Read CSV file with labels
		y_data = pd.DataFrame.from_csv(
			os.path.join(IMAGE_SOURCE, "..", "trainLabels.csv"))


	def __call__(self, X, y=None):
		self.X, self.y = X, y
		return self

	def gen(self):
		n_samples = self.X.shape[0]
		bs = self.batch_size

		for i in xrange((n_samples + bs - 1) // bs):

			start_index = i * bs
			end_index = (i+1) * bs

			indices = self.X[start_index:end_index]
			batch_keys = self.keys[indices]

			pipe = r.pipeline()

			y_batch = None

			if self.y is not None:
				y_batch = np.zeros((self.batch_size,))

			for i, key in enumerate(batch_keys):
				if self.y is not None:
					# Find `level` for `image_name` in trainLabels.csv file
					label = self.y_data.loc[key]['level']
					y_batch[i] = label

				pipe.get(key)

			X_batch = pipe.execute()
			X_batch = map(util.bin2array, X_batch)

			yield self.transform(X_batch, y_batch)

	def __iter__(self):
		import Queue
		queue = Queue.Queue(maxsize=4)
		sentinel = object()  # guaranteed unique reference

		# define producer (putting items into queue)
		def producer():
			for item in self.gen():
				queue.put(item)
				print "Queue size: %i" % (queue.qsize())
			queue.put(sentinel)

		# start producer (in a background thread)
		import threading
		thread = threading.Thread(target=producer)
		thread.daemon = True
		thread.start()

		# run as consumer (read items from queue, in current thread)
		item = queue.get()
		while item is not sentinel:
			yield item
			queue.task_done()
			item = queue.get()
			print "Queue size: %i" % (queue.qsize())

	def transform(self, Xb, yb):
		# Normalize
		Xbb = (Xb - self.mean) / self.std

		return Xbb, yb

class RedisIterator():
	def __init__(self, redis, keys):
		self.r = redis
		self.keys = keys
		self.index = 0

	def __iter__(self):
		while self.index < len(self.keys):
			_string = self.r.get(self.keys[self.index])
			_dat = util.bin2array(_string)
			yield _dat
			self.index += 1


class DataAugmentationBatchIterator(BatchIterator):
	"""
	Randomly changes images in the batch. Behaviour can be defined in params.py.
	Give mean and std of training set.
	"""
	def __init__(self, batch_size, mean, std):
		super(DataAugmentationBatchIterator, self).__init__(batch_size)
		self.mean = mean
		self.std = std

	def transform(self, Xb, yb):
		Xb, yb = super(DataAugmentationBatchIterator, self).transform(Xb, yb)

		Xbb = np.zeros((Xb.shape[0], Xb.shape[1], Xb.shape[2], Xb.shape[3]), dtype=np.float32)

		IMAGE_WIDTH = PIXELS
		IMAGE_HEIGHT = PIXELS

		random_flip = np.random.randint(2)

		def fast_warp(img, tf, output_shape=(PIXELS,PIXELS), mode='nearest'):
			"""
			This wrapper function is about five times faster than skimage.transform.warp, for our use case.
			"""
			return skimage.transform._warps_cy._warp_fast(img, tf.params, output_shape=output_shape, mode=mode)

		def random_perturbation_transform(zoom_range, rotation_range, shear_range, translation_range, do_flip=True):
			shift_x = np.random.uniform(*translation_range)
			shift_y = np.random.uniform(*translation_range)

			translation = (shift_x, shift_y)
			rotation = np.random.uniform(*rotation_range)
			shear = np.random.uniform(*shear_range)
			log_zoom_range = [np.log(z) for z in zoom_range]
			zoom = np.exp(np.random.uniform(*log_zoom_range))

			if do_flip and random_flip > 0: # flip half of the time
				shear += 180
				rotation += 180

			return build_augmentation_transform(zoom, rotation, shear, translation)

		center_shift = np.array((IMAGE_HEIGHT, IMAGE_WIDTH)) / 2. - 0.5
		tform_center = transform.SimilarityTransform(translation=-center_shift)
		tform_uncenter = transform.SimilarityTransform(translation=center_shift)

		def build_augmentation_transform(zoom=1.0, rotation=0, shear=0, translation=(0, 0)):
			tform_augment = transform.AffineTransform(scale=(1/zoom, 1/zoom),
													  rotation=np.deg2rad(rotation),
													  shear=np.deg2rad(shear),
													  translation=translation)
			tform = tform_center + tform_augment + tform_uncenter # shift to center, augment, shift back (for the rotation/shearing)
			return tform

		tform_augment = random_perturbation_transform(**AUGMENTATION_PARAMS)
		tform_identity = skimage.transform.AffineTransform()
		tform_ds = skimage.transform.AffineTransform()

		for i in range(Xb.shape[0]):
			Xbb[i, 0, :, :] = fast_warp(Xb[i][0], tform_ds + tform_augment + tform_identity, output_shape=(PIXELS,PIXELS), mode='nearest').astype('float32')
			if CHANNELS == 3:
				Xbb[i, 1, :, :] = fast_warp(Xb[i][1], tform_ds + tform_augment + tform_identity, output_shape=(PIXELS,PIXELS), mode='nearest').astype('float32')
				Xbb[i, 2, :, :] = fast_warp(Xb[i][2], tform_ds + tform_augment + tform_identity, output_shape=(PIXELS,PIXELS), mode='nearest').astype('float32')

			# Subtract mean and divide by std
			Xbb[i, :, :, :] -= self.mean
			Xbb[i, :, :, :] /= self.std

		return Xbb, yb
