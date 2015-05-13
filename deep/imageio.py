import numpy as np
import pandas as pd
from skimage.io import imread
from params import *
import glob
import os


class ImageIO():

    def _load_images_from_disk(self, image_type="train"):
        fnames = glob.glob(os.path.join(IMAGE_SOURCE, image_type, "*.jpeg"))

        fnames = fnames[:5000]

        # Read CSV file with labels
        y_data = pd.DataFrame.from_csv(
            os.path.join(IMAGE_SOURCE, "..", "trainLabels.csv"))

        n = len(fnames)

        # Create hdf5 file on disk
        f = h5py.File(IM2BIN_OUTPUT, "w")
        dsetX = f.create_dataset(
            "X_train", (n, CHANNELS, PIXELS, PIXELS), dtype=np.float32)
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
            image = imread(fileName, as_grey=False)
            # NOTE TO SELF: when using channels, dimshuffle from 01c to c01
            image = image.transpose(2, 0, 1)

            # Save image to hdf5 file
            dsetX[i] = image

            i += 1
            if i % 500 == 0:
                print "%.1f %% done" % (i * 100. / n)

        f.close()

    def im2bin_full(self):
        """
        Writes all images to a binary file.
        """
        # Write images to file
        self._load_images_from_disk(image_type="train")

        f = h5py.File(IM2BIN_OUTPUT, "r+")
        X = f['X_train']

        print "Computing mean and standard deviation..."

        # Compute mean and standard deviation and write to hdf5 file
        std, mean = np.sqrt(self.online_variance(X))

        meanSet = f.create_dataset("mean", mean.shape, dtype=np.float32)
        stdSet = f.create_dataset("std", std.shape, dtype=np.float32)

        meanSet[...] = mean
        stdSet[...] = std

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

    def online_variance(self, data):
        n = 0
        mean = 0.0
        M2 = 0
     
        for i, x in enumerate(data):
            n = n + 1
            delta = x - mean
            mean = mean + delta/n
            M2 = M2 + delta*(x - mean)

            if i % 1000 == 0:
                print "%.2f%%" % (float(i)/data.shape[0]*100)
     
        if n < 2:
            return 0
     
        variance = M2/(n - 1)
        return variance, mean

if __name__ == "__main__":
    ImageIO().im2bin_full()
