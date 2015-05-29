# -*- coding: utf-8 -*-
"""
Created on Thu Mar 05 20:42:04 2015

@author: Luc
"""
from __future__ import division
import kmeans
import numpy as np
import activationCalculation as act
import train_classifier as train
import predict_classifier as classifier
from sklearn.externals import joblib
import train_svc as svc
import h5py
import util
import scipy
from sklearn import metrics


def singlePipeline(nr_centroids, nr_it, 
                   label_path = "../data/preprocessed.h5", 
                   clsfr = "SGD", 
                   calc_centroids = True, 
                   dogfeed=True, 
                   train_model=True,
                   cache_size=4000,
                   degree=3,
                   tol=1e-3,
                   max_iter=-1,
                   kernel='rbf',
                   model_file='UNSPECIFIED'):
    
    
    
    if calc_centroids:
        print "calculating centroids..."
        #Finds the features using kmeans
        kmTrainer = kmeans.kMeansTrainer(nr_centroids = nr_centroids, nr_it = nr_it)    
        centroids = kmTrainer.fit()
        kmTrainer.save_centroids(centroids)
        
        print "calculating activations..."
        #Calculates the activaiton of the test set
        act_calc = act.ActivationCalculation()
        features = act_calc.pipeline(centroids)  
    else:
        print "loading activations from file..."
        #loads feature data
        feature_data = h5py.File("../data/activations_train/"+str(nr_centroids)+"activationkmeans.h5")
        features = feature_data["activations"]

    
    

    
    print "Loading labels from file..."
    #get the labels
    labels = util.load_labels(label_path)
    label_names = util.load_label_names(label_path)
    print "Got labels"

    
    
    if clsfr == "SGD": 
        if train_model:
            #Train the SGD classifier
            print "Begin training of SGD..."
            train.trainSGD(features, labels, nr_centroids)
            print "Training done"
        
        if not dogfeed:
            return
        
        print "Dogfeeding"
        #Predict based on SGD training
        print "Begin SGD predictions..."
        classified = classifier.predict(features, nr_centroids, degree=degree, cache_size=cache_size)
        print "Predicting done"        
        
    elif clsfr == "SVC" or clsfr == "NUSVR": 
        
        if train_model:
            print "Begin training of Model..."
            if clsfr=="SVC":
                #Train SVC classifier
                model = svc.train_svc(features, 
                                      labels, 
                                      nr_centroids,
                                      degree=degree,
                                      cache_size=cache_size,
                                      tol=tol,
                                      max_iter=max_iter,
                                      kernel = kernel)
            else :
                #Train SVC classifier
                model = svc.train_svc(features, 
                                      labels, 
                                      nr_centroids,
                                      degree=degree,
                                      cache_size=cache_size,
                                      tol=tol,
                                      max_iter=max_iter,
                                      kernel=kernel)    
            print "Training done"
        else:
            print "Loading model"
            model = joblib.load(model_file)
            
        
        if not dogfeed:
            return   
        
        print "Dogfeeding"
        #Predict based on SVC training
        print "Begin SVC predictions..."
        classified = model.predict_proba(features)
        print "Predicting done"
        
    
    else:
        print "Selected classifier not available, please use an available classifier"
        return
       
    
       
       
    print "Calculating log loss..."
    summing = 0
    correct = 0
    
    np.savetxt("meuk.csv", classified, delimiter=";")
    
    loss = metrics.log_loss(labels, classified)
    print loss

    print -np.mean(np.log(classified)[np.arange(len(labels)), labels])
    
    #calculate the log loss
    for i, label in enumerate(labels):
        
        actual = labels[i]   
        
        
        if(classified[i][label] == 0):
            summing+= np.log(10e-15)
        else:
            summing+= np.log(classified[i][label])
        if actual == np.argmax(classified[i]):
            correct += 1

    image = np.zeros((len(label_names),len(labels)))  
    
    for j, label_index in enumerate(labels):
        image[label_index,j] = 1

    scipy.misc.imsave('correct.png', image)
    scipy.misc.imsave('predicted.png', classified.T)
    
    error = image - classified.T
    
    scipy.misc.imsave('error.png', error)
    
    
    print "Calculation finished"  

    summing = -summing/len(labels)
    print "log loss: ", summing 
    print "correct/amount_of_labels: ", correct/len(labels)
    print "lowest classification score: ", np.min(classified)
   
#    print summing
    np.savetxt( "realLabel.csv", labels, delimiter=";")
   # np.savetxt( "SGD_label.csv", max_SGD, delimiter=";")  
    
    if calc_centroids is False:
        feature_data.close()       

    



if __name__ == '__main__':
    nr_centroids = 100  
    nr_it = 1           # Only used when calc_centroids is True
    clsfr = "SVC"       # Choice between SVC and SGD
    calc_centroids = True # Whether to calculate the centroids, 
                          # do NOT forget to set the nr_centroids to the desired centroidactivation file if False is selected.
    singlePipeline(nr_centroids=nr_centroids, nr_it=nr_it, clsfr=clsfr, calc_centroids = calc_centroids)
