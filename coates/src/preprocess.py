# -*- coding: utf-8 -*-

from __future__ import division
import os

import util
import imsquare
import impatch
import imutil
import re
import pickle


from scipy import misc
import numpy as np
import h5py

__PREPROCESS_VERSION__ = 2


"""
Preprocessing script

    1. Load all image paths into memory.
    2. Generate label tuple <classname (plankton type), filename, filepath>
    
    3. Determine mean, std and variance of all images
    4. Write labels to file
    
    5. For each image:
        1. Load image from disk
        2. Pad or stretch image into squar
        3. Resize (downsize probably) to common size
        4. Normalize image
        5. Patch image
        6. Flatten patches from 2D to 1D    
        7. Write the results to file
        
    6. Write metadata

"""

def preprocess(path='../data/train', 
               outpath="../data/preprocessed/image_stats.stat", **kwargs):
    """
    Preprocesses given folder, parameters (all optional):

    Args:
        path (str): Path to folder with retina images
        outpath (str): File to write to (.h5 file)
        patch_size (int): Width and length of patches
        image_size (int): Width and length to resize images to
        square_method (str): 'pad' or 'stretch', method to make images square.
    """    
    
    
    patch_size = kwargs.get('patch_size', 9)
    image_size = kwargs.get('image_size', 256)
        

    
    
    file_metadata, is_train = get_image_paths(path)   
    classnames, filenames, filepaths = zip(*file_metadata)  
    
   

    # Amount of images
    n = len(file_metadata)
    
    
    # Calculate some dimensions

    
    print "Patch size: {0}x{0} = {1}".format(patch_size, patch_size**2)
    print "Image size: {0}".format(image_size)
    print "Amount of images: {0}".format(n)

    
    
    metadata = {}
    metadata['patch_size'] = patch_size
    metadata['image_size'] = image_size
    metadata['image_count'] = n


    metadata['version'] = __PREPROCESS_VERSION__
    
    
    
    if preprocessing_is_already_done(outpath, metadata):
        print "----------------------------------------"
        return
    print "----------------------------------------"
    
    # Extract statistics such as the mean/std of image
    # Necessary for normalization of images

    mean_image, variance_image, std_image = extract_stats(filepaths, image_size)

    
    metadata['mean_image'] = mean_image 
    metadata['std_image' ] = std_image
    metadata['var_image' ] = variance_image
 
    print "---"

    
    print "Processing and writing..."
       

    
    print "Writing metadata (options used)" 
    pickle.dump(metadata, open(outpath, 'wb'))
    
  


def extract_stats(filepaths, image_size):
    print "Calculating mean, std and var of all images"
    
    #Running total (sum) of all images
    image_shape = (image_size, image_size, 3)
    count_so_far = 0
    mean = np.zeros(image_shape)
    M2 = np.zeros(image_shape)    
    
    
    n = len(filepaths)    
    
    for i, filepath in enumerate(filepaths):

        image = misc.imread(filepath)
        
        
        # Online statistics
        count_so_far = count_so_far+1
        delta = image - mean
        mean = mean + delta/count_so_far
        M2 = M2 + delta * (image-mean )
        if i % 50 == 0:
            util.update_progress(i/n)

    util.update_progress(1.0)
   
    mean_image = mean
    variance_image = 0
    variance_image = M2/(n-1)
    std_image = np.sqrt(variance_image)
   
    print "Plotting mean image (only shows afterwards)"
    util.plot(mean_image)
    return mean_image, variance_image, std_image
    
    
# Returns a dictionary from plankton name to index in ordered, unique set
# of plankton names
def gen_label_dict(classnames):
    unique_list = list(set(classnames));
    unique_list = sorted(unique_list)
    
    label_dict = {cname:i    for i, cname in enumerate(unique_list)}
    
    return label_dict

    
def write_labels(labels, h5py_file):
    h5py_file.create_dataset('labels', data=labels)
    
def write_label_names(label_names, h5py_file):
    h5py_file.create_dataset('label_names', data=label_names)
    
    

def write_metadata(dataset, metadata):

    for attr in metadata:
        dataset.attrs[attr] = metadata[attr]


def process(image, squarefunction, image_size):
    """
        Process a single image 
        - make horizontal by rotating 90 degrees if necessary
        - make square
        - resize
    """
    image = imutil.image_horizontal(image)
    image = squarefunction(image)
    image = imutil.resize_image(image, image_size)
    
    return image
    
def extract_patches(image, patch_size):
    """
     From image: extract patches, flatten patches
    """
    patches = impatch.patch(image, patch_size = patch_size)
    patches = [imutil.flatten_image(patch) for patch in patches]
    patches = np.array(patches)
    return patches
    
    
def preprocessing_is_already_done(filepath, metadata):
    print "----------------------------------------"
    print "Checking whether preprocess is already done for given settings"    
    
    if not os.path.exists(filepath):
        print "File {0} not found!".format(filepath)
        return False
    
    f = h5py.File(filepath)
    
    if not 'data' in f:
        print "Dataset not found in file"
        f.close()
        return False
        
    
    attrs = f['data'].attrs
    
    for key in metadata:
        
        inFile = attrs.get(key, None)
        inOptions = metadata.get(key, None)   
        
        if not inFile == inOptions:
            print "Found a different setting between file and given options"
            print "Key \"{0}\" has value \"{1}\" in file, and \"{2}\" in options".format(key, inFile, inOptions)
            f.close()
            return False
        
    print "Match between given options and data in file {0}".format(filepath)
    print "Not preprocessing again"
    f.close()
    return True


# Determines whether folder is train or test data
# Returns list of tuples of
# <classname of plankton, image filename, path to file>
#
# This classname is "UNLABELED" for test data
def get_image_paths(path):
    
    is_train = False
    
    for s in re.split('/', path):
        if s == 'train':
            is_train = True
            break
    
    return get_image_paths_test(path), is_train

def get_image_paths_test(path):
    metadata = []
    
    classname = "UNLABELED"

    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if filename != 'Thumbs.db':       
            metadata.append((classname, filename, filepath) )
        
    return metadata
        

def get_image_paths_train(path):
    
    metadata = []    
    
    # The classes are the folders in which the images reside
    classes = os.listdir(path)
    
    
    for classname in classes:
        for filename in os.listdir(os.path.join(path, classname)):
                filepath = os.path.join(path, classname, filename)
                if filename != 'Thumbs.db':      #ignore the thumbs.db file
                    metadata.append((classname, filename, filepath))
    
    return metadata


def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_numerical(paths):
    """ Sort the given list in the way that humans expect.
    """
    paths.sort(key=alphanum_key)
    return paths


if __name__ == '__main__':
    preprocess(path = '../data/trainrest')
