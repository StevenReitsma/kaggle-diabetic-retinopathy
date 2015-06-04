# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:59:17 2015

@author: Luc
"""
from __future__ import division
from sklearn.externals import joblib
import util

import time
import kmeans
import patch_reader
import pickle
import csv
import train_classifier
import random
import preprocess
import train_classifier
import numpy as np
import kappa

import activationCalculation

def train_centroids(ncentroids = 500):
    km_trainer = kmeans.kMeansTrainer(nr_centroids = ncentroids)
    start = time.time()

    batches = patch_reader.PatchReader(stride = 6)
    n_batches = batches.nbatches
    print "training centroids"
    util.update_progress(0)
    for i, (batch, key) in enumerate(batches):
        km_trainer.next_batch(batch)
        util.update_progress(i/n_batches)
    
    end = time.time()
    util.update_progress(1)
    print "Time elapsed: " + str((end-start))
    
    centroids = km_trainer.get_centroids()
    km_trainer.save_centroids(centroids)
    util.plot_centroids(centroids)
    return centroids

def get_activations(centroids):
    "calculating activations"
    ac = activationCalculation.ActivationCalculation()
    ac.pipeline(centroids)
    print "done"

def read_labels(file_path = '../data/trainLabels.csv'):
    """
    Randomizes order!
    """
    
    f = open(file_path)
    rows = []
    labels = []
    keys = []
    print "loading labels"
    try:
        reader = csv.reader(f)
        reader.next()
        for row in enumerate(reader):
            rows.append(row)
    finally:
        f.close()
        
    random.shuffle(rows)
    
    for row in rows:
        keys.append(row[1][0])
        labels.append(row[1][1])
        
    
    return labels, keys

def trainSGD(labels, keys, n_centroids):
    
    activations = load_activations(keys, n_centroids)
    print "training SGD"    
    train_classifier.trainSGD(activations, labels, n_centroids)
    
    

def cross_validate(labels, keys, n_centroids = 500):
    n_images = len(labels)
    nr_train = int(len(labels)*0.9)
    
    train_labels = labels[0:nr_train]
    train_keys = keys[0:nr_train]
    test_labels = labels[nr_train:n_images]
    test_keys = keys[nr_train:n_images]
    trainSGD(train_labels, train_keys, n_centroids)
    
    "Cross validating"
    test_features = load_activations(test_keys, n_centroids)
    clf = joblib.load('../models/sgd'+str(n_centroids)+'/classifier.pkl')
    predictions = clf.predict(test_features)
    n_correct = 0
    for i, p in enumerate(predictions):
        if p == test_labels[i]:
            n_correct+=1
        else:
            print "predicted: " + str(p) + " real label: " + test_labels[i]
    
    print "percent correct: " + str(n_correct/len(test_keys))
    print "Calcualting kappa: "
    kappa_score = kappa.quadratic_weighted_kappa(predictions, test_labels)
    print "Kappa: " + str(kappa_score)

    
    

def load_activations(keys, n_centroids):
    
    activations = np.zeros((len(keys), n_centroids*4))
    act_dict = pickle.load(open('../data/activations/activations500centroids.p', 'rb'))

    print "loading activations"    
    for i, key in enumerate(keys):
        activations[i] = act_dict[key]
    
    return activations
    
    
#    train_labels = 5

if __name__ == '__main__':
    labels, keys = read_labels()
    cross_validate(labels, keys)
    
    
    
#    dic = pickle.load(open('../data/activations/activations100centroids.p', 'rb'))
#    preprocess.preprocess(patch_size = 9)
#    centroids = train_centroids()
#    get_activations(centroids)
    
    