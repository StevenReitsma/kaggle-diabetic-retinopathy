# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:59:17 2015

@author: Luc
"""
from __future__ import division
import util
import impatch
import numpy as np
import time
import kmeans
import patch_reader
import activationCalculation

def train_centroids():
    km_trainer = kmeans.kMeansTrainer()
    start = time.time()

    batches = patch_reader.PatchReader()
    n_batches = batches.nbatches
    
    util.update_progress(0)
    for i, (batch, key) in enumerate(batches):
        km_trainer.next_batch(batch)
        util.update_progress(i/n_batches)
    
    end = time.time()
    util.update_progress(1)
    print "Time elapsed: " + str((end-start))
    
    return km_trainer.get_centroids()

def get_activations(centroids):
    ac = activationCalculation.ActivationCalculation()
    ac.pipeline(centroids)
    print "done"

if __name__ == '__main__':
    centroids = train_centroids()
    get_activations(centroids)
    
    