import numpy as np
import pandas as pd
from skimage.io import imread
from params import *
import glob
import os
from iterators import LevelDBIterator
import scipy

import leveldb
import util


class ImageIO():

    def _load_images_to_redis(self, db, image_type="train"):
        fnames = glob.glob(os.path.join(IMAGE_SOURCE, image_type, "*.jpeg"))

        n = len(fnames)

        image_names = []

        i = 0
        for fileName in fnames:
            # Get label and add to hdf5 file

            # Take filename and remove jpeg extension
            image_name = fileName.split('/')[-1][:-5]
            image_names.append(image_name)

            image = imread(fileName, as_grey=False)
            # NOTE TO SELF: when using channels, dimshuffle from 01c to c01
            image = image.transpose(2, 0, 1)

            # Save image to hdf5 file
            db.Put(image_name, image.tobytes())

            i += 1
            if i % 100 == 0:
                print "%.1f %% done" % (i * 100. / n)
        return image_names


    def im2bin_full(self):
        """
        Writes all images to a binary file.
        """
        db_train = leveldb.LevelDB('./dbtrain')

        # Write images to file
        keys = self._load_images_to_redis(db_train, image_type="train")

        print "Computing mean and standard deviation..."

        # Compute mean and standard deviation and write to hdf5 file
        var, mean = self.online_variance(db_train, image_type="train", keys=keys)
        std = np.sqrt(var)

        db_train.Put('mean', mean.tobytes())
        db_train.Put('std', std.tobytes())

        print "Done!"

    def load_train_full(self):
        f = h5py.File(IM2BIN_OUTPUT, "r")
        X = f['X_train']
        y = f['y_train']

        # Load full data set to memory, should be changed to a disk stream
        # later
        X = X[()].astype(np.float32, copy=False)
        y = y[()].astype(np.int32, copy=False)

        return X, y

    def load_mean_std(self, r):
        mean = r.get('mean')
        std = r.get('std')

        return mean, std

    def online_variance(self, db, image_type, keys):
        n = 0
        mean = 0.0
        M2 = 0

        data = LevelDBIterator(r, keys)

        for i, x in enumerate(data):
            n = n + 1
            delta = x - mean
            mean = mean + delta/n
            M2 = M2 + delta*(x - mean)

            if i % 100 == 0:
                print "%.2f%%" % (float(i)/len(keys)*100)

        if n < 2:
            return 0

        variance = M2/(n - 1)
        return variance, mean


    def calc_variance(self, image_type="train"):
        fnames = glob.glob(os.path.join(IMAGE_SOURCE, image_type, "*.jpeg"))

        nl = len(fnames)

        n = 0
        mean = 0.0
        M2 = 0

        i = 0
        for fileName in fnames:

            image = imread(fileName, as_grey=False)

            n = n + 1
            delta = image - mean
            mean = mean + delta/n
            M2 = M2 + delta*(image - mean)


            i += 1
            if i % 100 == 0:
                print "%.1f %% done" % (i * 100. / nl)

        if n < 2:
            return 0,image

        variance = M2/(n - 1)
        std = np.sqrt(variance)

        scipy.misc.imsave('../data/processed/std.png', std)
        scipy.misc.imsave('../data/processed/mean.png', mean)

        return variance, mean




if __name__ == "__main__":
    var, mean = ImageIO().calc_variance()
