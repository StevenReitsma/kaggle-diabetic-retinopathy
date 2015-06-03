import numpy as np
from nolearn.lasagne import BatchIterator
from params import *
import scipy
from augment import Augmenter
import cv2

from multiprocessing import Process, Queue, JoinableQueue, Value
from threading import Thread

import util
import time

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

	def gen(self, indices):

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

			return {'input': X_batch, 'coates': coates_batch}, y_batch
		else:
			return X_batch, y_batch

	def __iter__(self):
		queue = JoinableQueue(maxsize=params.N_PRODUCERS*2)

		n_batches, job_queue = self.start_producers(queue)

		# Run as consumer (read items from queue, in current thread)
		for x in xrange(n_batches):
			item = queue.get()
			#print len(item[0]), queue.qsize(), "GET"
			yield item
			queue.task_done()

		#queue.join() #Lock until queue is fully done
		queue.close()
		job_queue.close()



	def start_producers(self, result_queue):
		jobs = Queue()
		n_workers = params.N_PRODUCERS
		batch_count = 0

		#Flag used for keeping values in queue in order
		last_queued_job = Value('i', -1)

		for job_index, batch in enumerate(util.chunks(self.X,self.batch_size)):
			batch_count += 1
			jobs.put( (job_index,batch) )

		# Define producer (putting items into queue)
		def produce(id):
			while True:
				job_index, task = jobs.get()

				if task is None:
					#print id, " fully done!"
					break

				result = self.gen(task)

				while(True):
					#My turn to add job done
					if last_queued_job.value == job_index-1:

						with last_queued_job.get_lock():
							result_queue.put(result)
							last_queued_job.value += 1
							#print id, " worker PUT", job_index
							break

		#Start workers
		for i in xrange(n_workers):

			if params.MULTIPROCESS:
				p = Process(target=produce, args=(i,))
			else:
				p = Thread(target=produce, args=(i,))

			p.daemon = True
			p.start()

		#Add poison pills to queue (to signal workers to stop)
		for i in xrange(n_workers):
			jobs.put((-1,None))


		return batch_count, jobs


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
		self.center_shift = np.array((params.PIXELS, params.PIXELS)) / 2. - 0.5
		self.i = 0

		self.rotations = [0, 45, 90, 135, 180, 225, 270, 315]
		self.flips = [True, False]
		self.hue = [0]
		self.saturation = [0]

		self.ttas = len(self.rotations) * len(self.flips) * len(self.hue) * len(self.saturation)

	def transform(self, Xb, yb):
		if params.MULTIPROCESS:
			print "Batch %i/%i" % (self.i, self.X.shape[0]/self.batch_size/params.N_PRODUCERS)
		else:
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
