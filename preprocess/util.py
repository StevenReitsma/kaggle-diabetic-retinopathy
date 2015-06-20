# -*- coding: utf-8 -*-
import h5py
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import math

def load_unsupervised(filepath="../data/preprocessed.h5", shuffle=True):
    f = h5py.File(filepath)
    dset = f["data"]    
   # rdata = []
    dimensions = (len(dset), len(dset[0]))
    #dimensions = len(dset)
    rdata = np.zeros(dimensions, dtype=np.float32)
    
    dset.read_direct(rdata)
    
    if shuffle:
        np.random.shuffle(rdata)
    
    f.close()    
    
    return rdata;

def plot(image, invert = False, cmap=plt.cm.binary):
    
    if invert:
        image = np.ones(len(image)) - image
    
    plt.gray()
    plt.imshow(image, cmap=cmap)


# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
# From http://stackoverflow.com/questions/3160699/python-progress-bar
def update_progress(progress):
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "="*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()
    
