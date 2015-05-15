import numpy as np
import pandas as pd
from skimage.io import imread
from params import *
import glob
import os
from iterators import RedisIterator

from StringIO import StringIO
import redis

import util


class ImageIO():

    def _load_images_to_redis(self, image_type="train"):
        fnames = glob.glob(os.path.join(IMAGE_SOURCE, image_type, "*.jpeg"))

        fnames = fnames[:1000]

        n = len(fnames)

        if image_type=="train":
            db = 0
        else:
            db = 1

        r = redis.StrictRedis(db=db)

        image_names = []

        i = 0
        for fileName in fnames:
            # Get label and add to hdf5 file

            # Take filename and remove jpeg extension
            image_name = fileName.split('/')[-1][:-5]
            image_names.append(image_name)

            # Read in the image as grey-scale
            image = imread(fileName, as_grey=False)
            # NOTE TO SELF: when using channels, dimshuffle from 01c to c01
            image = image.transpose(2, 0, 1)

            # Save image to hdf5 file
            r.set(image_name, image.tobytes())

            i += 1
            if i % 100 == 0:
                print "%.1f %% done" % (i * 100. / n)
        return image_names


    def im2bin_full(self):
        """
        Writes all images to a binary file.
        """
        # Write images to file
        keys = self._load_images_to_redis(image_type="train")

        print "Computing mean and standard deviation..."

        # Compute mean and standard deviation and write to hdf5 file
        var, mean = self.online_variance(image_type="train", keys=keys)
        std = np.sqrt(var)

        r = redis.StrictRedis(db=0)

        r.set('mean', mean.tobytes())
        r.set('std', std.tobytes())

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

    def get_hdf5_train_stream(self):
        f = h5py.File(IM2BIN_OUTPUT, "r")
        X = f['X_train']
        y = f['y_train']

        return X, y

    def load_mean_std(self):
        f = h5py.File(IM2BIN_OUTPUT, "r")
        mean = f['mean']
        std = f['std']

        return mean, std

    def online_variance(self, image_type, keys):

        db = 0 if image_type == 'train' else 1
        r = redis.StrictRedis(db=db)

        n = 0
        mean = 0.0
        M2 = 0


        redisData= RedisIterator(r, keys)

        for i, x in enumerate(redisData):
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

if __name__ == "__main__":
    ImageIO().im2bin_full()
