# -*- coding: utf-8 -*-
import random
import preprocess as pre
from scipy import misc
import numpy as np

def patch(image, n=0, patch_width=6):    
    """
       Patches an image (samples sub-images)
    """
    
    patches = []
    
    xlength = len(image[0])
    ylength = len(image)
    patch_size = patch_width**2*3    
    
    if patch_width > xlength or patch_width > ylength:
        raise Exception("Patchsize too big for given image")


    # Max top left index from which patches are taken        
    xindexmax = xlength - patch_width    
    yindexmax = ylength - patch_width 

    
    nmaxpatches = (xindexmax+1) * (yindexmax+1)
    
    
    
    if n > nmaxpatches:
        raise Exception("Impossible to extract this many patches from image")
        
    if n == 0:
        n = nmaxpatches
        
    coords = [(x,y) for x in range(xindexmax+1) for y in range(yindexmax+1)]
    patches = np.zeros((n,patch_size))
    
    # Shuffle list of coords
    #random.shuffle(coords)
    
    
    for i, coord in enumerate(coords):
        if i >= n:
            break
        
        x, y = coord

        patch = image[x:(x+patch_width),y:(y+patch_width)]
        np_patch = np.reshape(patch, (patch_size))
        patches[i] = np_patch
    
    return patches
    
    
def npatch(imagesize, patchsize):
    return (imagesize + 1 - patchsize)**2
    """
        Maximum amount of patches extracted from image given size
    """
    
   
         
    