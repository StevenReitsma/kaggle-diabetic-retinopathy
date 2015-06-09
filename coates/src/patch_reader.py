# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:10:11 2015

@author: Luc
"""

from __future__ import division
import os
import math
import impatch
from scipy import misc
import pickle
import imutil
import util
import time
import numpy as np

class PatchReader(object):
    """"
    Online patchreader
    """
    
    def __init__(self, filepath = '../data/train', stats_path = '../data/preprocessed/image_stats.stat', batch_size = -1, stride = 1):
        image_stats = pickle.load(file(stats_path, 'rb'))
        self.image_size = image_stats['image_size']
        self.patch_size = image_stats['patch_size']
        self.mean_image = image_stats['mean_image']
        self.std_image = image_stats['std_image']
        self.files = os.listdir(filepath)
        self.dir_path = filepath + '/'
        self.stride = stride
        self.patchert = impatch.ImPatch(image_shape = (self.image_size, self.image_size), patch_width = self.patch_size, stride = self.stride)
        self.n_patches_image = self.patchert.nmaxpatches #n patches per image

        if batch_size == -1:
            self.batch_size = self.n_patches_image
        else:   
            self.batch_size = batch_size
        
        self.nbatches = math.ceil(len(self.files)*self.n_patches_image/self.batch_size)
        
        self.current = 0 
        
    
    def __iter__(self):
        return self
    
    def next(self):
        if(self.current >= self.nbatches):
            raise StopIteration
        else:
            image_path = self.files[self.current]
            image = misc.imread(self.dir_path + image_path)
            image = imutil.normalize(image, self.mean_image, self.std_image)
            patches = self.patchert.patch(image)
            

            
            for i, patch in enumerate(patches):
                mean = np.mean(patch)
                std = np.std(patch)
                patches[i] = imutil.normalize(patch, mean, std)
#                util.update_progress(i/len(patches))
            key = self.get_key(image_path)
            self.current+=1
            
            return patches, key

    def get_key(self, name):
        return name.split('.')[0]

if __name__ == '__main__':
    for i, (patches, key) in enumerate(PatchReader(stride = 2)):
        print i, key, len(patches)
        


        