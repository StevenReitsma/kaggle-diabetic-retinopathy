import numpy as np
from skimage.io import imread
from params import *
import glob
import os

import cv2
import scipy
import scipy.misc

class ImageIO():

    def load_mean_std(self, circularized=False):


        if circularized:
            mean = np.load(params.IMAGE_SOURCE + '/mean_c.npy').transpose(2,0,1).astype(np.float32)
            std = np.load(params.IMAGE_SOURCE + '/std_c.npy').transpose(2,0,1).astype(np.float32)
        else:
            mean = np.load(params.IMAGE_SOURCE + '/mean.npy').transpose(2,0,1).astype(np.float32)
            std = np.load(params.IMAGE_SOURCE + '/std.npy').transpose(2,0,1).astype(np.float32)


        return mean, std


    def calc_variance(self, image_type="train"):
        fnames = glob.glob(os.path.join(params.IMAGE_SOURCE, image_type, "*.jpeg"))

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

        np.save(params.IMAGE_SOURCE + '/std.npy', std)
        np.save(params.IMAGE_SOURCE + '/mean.npy', mean)

        scipy.misc.imsave('mean.png', np.cast['int32'](mean))
        scipy.misc.imsave('std.png', np.cast['int32'](std))


        return variance, mean


    def circularize_mean_std(self):
        """
            - takes the mean and std image
            - rotates it all possible degrees and flips
            - calculates the mean
            - saves to disk
        """

        mean, std = self.load_mean_std(circularized=False)
        mean = mean.transpose(1, 2, 0)
        std = std.transpose(1,2,0)

        means = []
        stds = []

        pix = mean.shape[0]

        center_shift = np.array((pix,pix)) / 2. - 0.5

        for rotation in xrange(360):
            M = cv2.getRotationMatrix2D((center_shift[0], center_shift[1]), rotation, 1.0)

            m_rot = cv2.warpAffine(mean, M, (pix, pix))
            s_rot = cv2.warpAffine(std, M, (pix, pix))

            m_rot_flipped = cv2.flip(m_rot, 0)
            s_rot_flipped = cv2.flip(s_rot, 0)

            means.append(m_rot)
            stds.append(s_rot)

            means.append(m_rot_flipped)
            stds.append(s_rot_flipped)


        mean_circularized = sum(means) / len(means)
        std_circularized = sum(stds) / len(stds)

        np.save(params.IMAGE_SOURCE + '/mean_c.npy', mean_circularized)
        np.save(params.IMAGE_SOURCE + '/std_c.npy', std_circularized)

        scipy.misc.imsave('mean_c.png', np.cast['int32'](mean_circularized))
        scipy.misc.imsave('std_c.png', np.cast['int32'](std_circularized))


if __name__ == "__main__":
    var, mean = ImageIO().calc_variance()
    ImageIO().circularize_mean_std()
