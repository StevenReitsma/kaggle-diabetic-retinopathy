# -*- coding: utf-8 -*-

from sklearn import svm
from sklearn.externals import joblib
import os

def train_svc(samples, labels, nr_centroids, degree=3, cache_size=4000, max_iter = -1, tol = 1e-3, kernel='rbf'):
    clf = svm.SVC(degree=degree, 
                  cache_size = cache_size, 
                  probability = True, 
                  verbose = 1,
                  tol = tol,
                  max_iter = max_iter,
                  kernel = kernel)
    clf.fit(samples, labels)
    
    file_path = '../models/svc' + str(nr_centroids) + '/'
    if not os.path.exists(file_path):
              os.makedirs(file_path)
          
    joblib.dump(clf, file_path + '/classifier.pkl')
    return clf 
