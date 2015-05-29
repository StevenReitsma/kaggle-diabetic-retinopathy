# -*- coding: utf-8 -*-

from sklearn.linear_model import SGDClassifier
import numpy as np
import preprocess
import kmeans
import activationCalculation as ac
import kmeans_runner
import predict_classifier as pc
import util
import h5py
import csv
import batchreader as br
#
# This file contains all steps for training k-means with a SVC classifier.
#

#----------------------------------------------------
#OPTIONS

#Classifier
classifier="SGD" #'SGD', 'SVC', 'NUSVR', only tested for SVC

# NUSVR does not seem to work :c
# Can output likelihood-order, but not probabilities, meh.

#Preprocessing options
square_method = 'pad' #Either 'pad' or 'stretch'
patch_size = 6
image_size = 32 #Common size for all images to be resized to

train_folder = '../data/train/'
test_folder = '../data/test/'

processed_train_filename = '../data/preprocessed/preprocessed_train.h5'
#processed_test_filename = '../data/preprocessed_test.h5'
processed_test_filename = '../data/preprocessed_test.h5'



#K-Means options
nr_iterations = 10
nr_centroids = 500
centroids_folder = "../data/centroidskmeans/"
activations_folder_test = "../data/activations_test/"
activations_folder_train = "../data/activations_train/"

#Misc options
nr_pool_regions = 4

#Classifier options
degree = 3
cache_size = 6000 #In MB
max_iter = 5000
tol = 1e-3
kernel = 'poly'


#----------------------------------------------------
# PREPROCESS
# Filesize with default settings above 2.96GB and 12.7GB

def one():
    #Train images, takes 5 min
    preprocess.preprocess(path=train_folder, 
                          outpath=processed_train_filename, 
                          patch_size=patch_size,
                          image_size=image_size)

# You can do this after step 6
def two():
    #Test images, takes 30 min+
    preprocess.preprocess(path=test_folder, 
                          outpath=processed_test_filename, 
                          patch_size=patch_size,
                          image_size=image_size,
                          train_data_file=processed_train_filename)

#----------------------------------------------------
# K-MEANS 

# CREATE CENTROIDS
# 30 min with default settings

def three():
    km_trainer = kmeans.kMeansTrainer(nr_centroids = nr_centroids, 
                                      nr_it = nr_iterations)
    print "Creating centroids"
    centroids = km_trainer.fit()
    print "Saving centroids to file"
    km_trainer.save_centroids(centroids, 
                              file_path=centroids_folder)
    print "Done"
    
    
# ACTIVATIONS OF TRAIN SET
# 5 min
def four():
    calculator = ac.ActivationCalculation()
    
    km = kmeans.kMeansTrainer()
    print "Loading centroids"
    centroids = km.get_saved_centroids(nr_centroids, 
                                       file_path=centroids_folder)
    
    print "Calculating activations of train data"
    calculator.pipeline(centroids, 
                        n_pool_regions = nr_pool_regions,
                        file_path = activations_folder_train,
                        data_file = processed_train_filename)
    print "Done"

#----------------------------------------------------
# TRAIN CLASSIFIER (SVC, NUSVR OR SGD)
def five():
    kmeans_runner.singlePipeline(nr_centroids, 
                                 nr_iterations, 
                                 label_path = processed_train_filename,
                                 clsfr = classifier,
                                 calc_centroids = False,
                                 dogfeed = False,
                                 train_model = True,
                                 cache_size = cache_size,
                                 degree = degree,
                                 tol = tol,
                                 max_iter = max_iter)

#!!!
# OPTIONAL, Predict train data to see some performance measure
# Useful to get estimate of performance before creating submission
#!!!

def six():
    
    model_filename = '../models/'+classifier.lower()+str(nr_centroids)+'/classifier.pkl'
    #Also creates nice visualizations of predictions as png files
    #in src folder
    kmeans_runner.singlePipeline(nr_centroids, 
                                 nr_iterations, 
                                 label_path = processed_train_filename,
                                 clsfr = classifier,
                                 calc_centroids = False,
                                 dogfeed = True,
                                 train_model = False,
                                 model_file = model_filename)
    
    
#----------------------------------------------------
# CALCULATE ACTIVATIONS OF TEST DATA
def steven():
    calculator = ac.ActivationCalculation()
    
    km = kmeans.kMeansTrainer()
    print "Loading centroids"
    centroids = km.get_saved_centroids(nr_centroids, 
                                       file_path=centroids_folder)
    print "Calculating activations of test data"
    calculator.pipeline(centroids, 
                        n_pool_regions = nr_pool_regions,
                        file_path = activations_folder_test,
                        data_file = processed_test_filename)
    print "Done"
    
    
    
#----------------------------------------------------
# USE TRAINED MODEL TO PREDICT TEST SETS

def eight():
    model_filename = '../models/'+classifier.lower()+str(nr_centroids)+'/classifier.pkl'
    

    pc.predict_classifier(model=model_filename,
                          activations_folder = activations_folder_test,
                          nr_centroids=nr_centroids)
    print "Done"

def test():
     meta = util.load_metadata(filepath="../data/preprocessed/preprocessed_train.h5")
     batch_size = meta['patches_per_image']
     batches = br.BatchReader(filepath="../data/preprocessed/preprocessed_train.h5", batchsize=batch_size)
     batch = batches.next()
     print batch

test()