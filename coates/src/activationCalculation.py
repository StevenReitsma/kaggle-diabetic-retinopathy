# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 21:49:01 2015

@author: Luc and Tom
"""
from __future__ import division
from pooling import pool
import numpy as np
import kmeans
import os
import util
import patch_reader
import matplotlib.pyplot as plt
import pickle


class ActivationCalculation():
            
        
    def distance_to_centroids(self, patches, centroids):
        # Triangle (soft) activation function
        pp = np.sum(np.square(patches), axis=1) # Dot product patchesxpatches\
        cc = np.sum(np.square(centroids), axis=1) # Dot product centroidsxcentroids
        pc = 2*np.dot(patches, centroids.T) # 2* Dot product patchesxcentroids

        z = np.sqrt(cc + (pp - pc.T).T) # Distance measure
        mu = z.mean(axis=0)
        activation = np.maximum(0, mu-z) # Similarity measure

        return activation
    
 
    def _distance_to_centroids(self, patches, centroids):
        #self.visualize_activation(centroids)
        
        activations = np.zeros((patches.shape[0],centroids.shape[0]) )
        
        for i, patch in enumerate(patches):
            for j, centroid in enumerate(centroids):

                #print "Centroid/patch"
                #plt.imshow(centroid.reshape(6, 6), interpolation='nearest')
                #plt.show()
                #plt.imshow(patch.reshape(6,6), interpolation='nearest')
                #plt.show()
                
                act = np.square(centroid - patch)
                
                #print "Activation"
                #plt.imshow(act.reshape(6,6), interpolation='nearest')
                #plt.show()
                
                #print i, j
                #print np.max(centroid), np.max(patch)
               # print np.mean(centroid), np.mean(patch)

                
                activations[i, j] = np.sum(act)
            
        return activations
        
        
    
    def normalize(self, activations, activations_dict, key_list):
        std = np.std(activations, axis = 0)
        mean = np.mean(activations, axis = 0)
        
        for key in key_list:
            activations_dict[key] = (np.cast['float32']((activations_dict[key]-mean)/std))

        
        
        return activations
  
    
    
    def pipeline(self, centroids, file_path = "../data/activations/", batch_size = -1, n_pool_regions = 4):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        
        patches = patch_reader.PatchReader(stride = 3)
        
       

        dimensions = (patches.nbatches , len(centroids)*n_pool_regions) # Set dimensions to #imagesx4*#centroids
        activations_dict = {}
        activations = np.zeros(dimensions)
        key_list = []
        
        for i, (batch, key) in enumerate(patches):
            activation = self.distance_to_centroids(batch, centroids) # Calculate activations for each patch to each centroid        
            
            pooled = pool(activation, n_pool_regions = n_pool_regions) # Returns a vector with length 4x#centroids
            activations_dict[key] = pooled
            activations[i] = pooled
            key_list.append(key)
            util.update_progress(i/patches.nbatches)
            
        util.update_progress(1)
       
        print "Normalizing activations..."
        activations = self.normalize(activations, activations_dict, key_list)
        print "Normalizing done"
        print "Write to file"        
        pickle.dump(activations_dict, open('../data/activations/activations' + str(len(centroids)) + 'centroids.p', "wb"))
        
        return activations

        
        
        
    def visualize_activation(self, activations):
        print activations.shape
        patch_size = np.sqrt(activations.shape[1])
        n_features = activations.shape[0]
    
        # Reshape to 2D slabs
        reshaped = np.reshape(activations, (n_features, patch_size, patch_size))
        

        length = int(np.sqrt(reshaped.shape[0]))
        
        f, ax = plt.subplots(length, length)
        
        for i in range(0, length):
            for j in range(0, length):
             ax[i, j].imshow(reshaped[i*length+j], cmap = 'Greys', interpolation = 'nearest')
             ax[i, j].axis('off')
        
        plt.show()
        
        
    def visualize_activation_alt(self, activations):
        patch_size = np.sqrt(activations.shape[0])
        
        one = activations[:,0]
        
        im = np.reshape(one, (patch_size,patch_size))

        plt.imshow(im, cmap='Greys', interpolation= 'nearest')        
        
        plt.show()

    
    
if __name__ == '__main__':
    km = kmeans.kMeansTrainer()
    centroids = km.get_saved_centroids(100)
    #util.plot_centroids(centroids, "../data/centroidskmeans")
    sup_km = ActivationCalculation()
    sup_km.pipeline(centroids = centroids, data_file="../data/preprocessed_test.h5")
    
    


