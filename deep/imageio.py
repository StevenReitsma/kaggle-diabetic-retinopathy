import numpy as np
from skimage.io import imread
from params import *
import glob
import os


class ImageIO():

    def load_mean_std(self):
        mean = np.load(IMAGE_SOURCE + '/mean.npy').transpose(2,0,1).astype(np.float32)
        std = np.load(IMAGE_SOURCE + '/std.npy').transpose(2,0,1).astype(np.float32)

        return mean, std


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

        np.save(IMAGE_SOURCE + '/std.npy', std)
        np.save(IMAGE_SOURCE + '/mean.npy', mean)

        return variance, mean


if __name__ == "__main__":
    var, mean = ImageIO().calc_variance()
