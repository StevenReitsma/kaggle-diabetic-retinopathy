import sys
sys.path.append("./../deep")
from params import *
from augment import Augmenter
import util

import scipy.misc
import numpy as np
from multiprocessing import Pool

import timeit


def read_and_augment(keys):
    augmenter = Augmenter()

    images = np.zeros( (len(keys),CHANNELS,PIXELS,PIXELS), dtype=np.float32)
    for i, key in enumerate(keys):
        image = scipy.misc.imread(IMAGE_SOURCE + "/" + 'train' + "/" + key + ".jpeg").transpose(2, 0, 1)
        image = image/256.0
        images[i] = image

    return augmenter.augment(images)

def augment_multithreaded(key_batches, n_threads=4):
        p = Pool(n_threads)
        augmented = p.map(read_and_augment, key_batches)
        augmented = np.array(augmented)

def augment_singlethreaded(key_batches):
        augmented = map(read_and_augment, key_batches)
        augmented = np.array(augmented)

def profile(subset=1000, multi=True, n_threads = 4, batch_size=64):

    # Load a bunch of imagenames
    y = util.load_labels()
    y = y[:subset]
    keys = y.index.values

    #Create sublists (batches)
    batched_keys = util.chunks(keys, batch_size)

    if multi:
        augment_multithreaded(batched_keys, n_threads=n_threads)
    else:
        augment_singlethreaded(batched_keys)

if __name__ == '__main__':
    print timeit.timeit('ma.profile(multi=True,n_threads=4)',setup='import multithread_augment as ma',number=10)
    #print timeit.timeit('ma.profile(multi=False,n_threads=4)',setup='import multithread_augment as ma',number=10)
