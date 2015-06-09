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


def flatten(collection):
    return [item for sublist in collection for item in sublist]
    
def plot_mean_image():
    plot_meta_image('mean_image', invert=True, cmap=plt.cm.binary)

def plot_std_image():
    plot_meta_image('std_image')
    
def plot_var_image():
    plot_meta_image('var_image')

def plot(image, cmap=plt.cm.binary):
    
    plt.imshow(image, cmap=cmap)

    
def plot_meta_image(attr_name, invert = False, cmap=None):
    meta = load_metadata()
    im = meta[attr_name]
    
    plot(im, invert, cmap) 
    
    

def plot_centroids(centroids, file_path = '../data/centroids/plots'):
    file_path = file_path+str(len(centroids))+'/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    print "start plotting"        
    for i, centroid in enumerate(centroids):
        update_progress(i/len(centroids))
        im_size = np.sqrt(len(centroids[0])/3)
        centroid_matrix = np.reshape(centroid, (im_size,im_size,3))
        centroid_matrix = np.uint8(centroid_matrix)
        plt.gray()
        plt.imsave(file_path + "centroid" + str(i) + ".png", centroid_matrix)
         
    update_progress(1.0)
    print "plotting done"

def denormalize_centroids(centroids):
    min_val = np.min(centroids)
    centroids+= abs(min_val)
    factor = 255/np.max(centroids)
    centroids = centroids*factor
    return centroids
    
        
    
    
def load_metadata(filepath="../data/preprocessed/preprocessed_train.h5"):
    f = h5py.File(filepath)
    dset = f["data"]  
    
    attributes = {}
    
    # Copy into an in-memory dict
    for key, val in dset.attrs.items():
        attributes[key] = val
    
    f.close()
    return attributes

def load_labels(filepath="../data/preprocessed.h5"):
    f = h5py.File(filepath)
    dset = f["labels"]
    labels = [label for label in dset]
    f.close()
    return labels
    
def load_label_names(filepath="../data/preprocessed.h5"):
    f = h5py.File(filepath)
    dset = f["label_names"]
    names = [name for name in dset]
    f.close()
    return names

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
    
