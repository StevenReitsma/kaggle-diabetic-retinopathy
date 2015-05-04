import numpy as np
import pandas as pd
from skimage.io import imread, imsave
from sklearn import preprocessing
import h5py
from params import *
from sklearn.utils import shuffle
from augmenter import Augmenter
import glob
import os
import matplotlib.pyplot as plt

class ImageIO():
	def _load_images_from_disk(self, image_type = "train"):
		imageSize = CHANNELS * PIXELS * PIXELS
		fnames = glob.glob(os.path.join(IMAGE_SOURCE, image_type, "*.jpeg"))

		# Take subset
		fnames = fnames[:7500]

		# Read CSV file with labels
		y_data = pd.DataFrame.from_csv(os.path.join(IMAGE_SOURCE, "..", "trainLabels.csv"))

		n = len(fnames)

		# Create hdf5 file on disk
		f = h5py.File(IM2BIN_OUTPUT, "w")
		dsetX = f.create_dataset("X_train", (n, CHANNELS, PIXELS, PIXELS), dtype=np.float32)
		dsety = f.create_dataset("y_train", (n,), dtype=np.int32)

		i = 0
		for fileName in fnames:
			# Get label and add to hdf5 file

			# Take filename and remove jpeg extension
			image_name = fileName.split('/')[-1][:-5]
			# Find `level` for `image_name` in trainLabels.csv file
			label = y_data.loc[image_name]['level']
			dsety[i] = label

			# Read in the image as grey-scale
			image = imread(fileName, as_grey=True)
			# NOTE TO SELF: when using channels, dimshuffle from 01c to c01

			# Save flattened image to hdf5 file
			dsetX[i, :, :, :] = image
			
			i += 1
			if i % 500 == 0:
				print "%.1f %% done" % (i * 100. / n)

	def im2bin_full(self):
		"""
		Writes all images to a binary file.
		"""
		# Write images to file
		self._load_images_from_disk(image_type = "train")

		f = h5py.File(IM2BIN_OUTPUT, "r+")
		X = f['X_train']

		print "Computing mean and standard deviation..."

		# Compute mean and standard deviation and write to hdf5 file
		mean = np.mean(X, axis=0)
		std = np.std(X, axis=0)

		meanSet = f.create_dataset("mean", mean.shape, dtype=np.float32)
		stdSet = f.create_dataset("std", std.shape, dtype=np.float32)

		meanSet[...] = mean
		stdSet[...] = std

		print "Done!"

	def load_train_full(self):
		f = h5py.File(IM2BIN_OUTPUT, "r")
		X = f['X_train']
		y = f['y_train']

		# Load full data set to memory, should be changed to a disk stream later
		X = X[()].astype(np.float32, copy=False)
		y = y[()].astype(np.int32, copy=False)

		return X,y

	def get_train_stream(self):
		f = h5py.File(IM2BIN_OUTPUT, "r")
		X = f['X_train']
		y = f['y_train']

		return X, y

	def load_mean_std(self):
		f = h5py.File(IM2BIN_OUTPUT, "r")
		mean = f['mean']
		std = f['std']

		return mean, std

if __name__ == "__main__":
	ImageIO().im2bin_full()
