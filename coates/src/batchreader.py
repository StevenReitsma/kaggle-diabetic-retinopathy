# -*- coding: utf-8 -*-

from __future__ import division
import h5py
import math


#Usage:
#
# for chunk_of_data in BatchReader():
#     print chunk_of_data
#
# In the last iteration the remainder chunk is returned
# (which may be smaller than batchsize)

class BatchReader(object):
    
    def __init__(self, filepath="../data/preprocessed/preprocessed_train.h5", batchsize=10000, dataset="data"):
        self.path = filepath;
        self.batch_size = batchsize
        
        self.file = h5py.File(filepath)
        self.dset = self.file[dataset]    
        self.dimensions = (len(self.dset), len(self.dset[0]))    
        

        # Iteration index
        self.current = 0
        # Max iterations
        self.nbatches = math.ceil(self.dimensions[0]/self.batch_size)

            
    def __iter__(self):
        return self
        
    def next(self):
        if self.current >= self.nbatches:
            self.file.close()
            raise StopIteration
        else:
            fromIndex = self.current*self.batch_size
            toIndex = fromIndex + self.batch_size
            
            dat = self.dset[fromIndex : toIndex]

            self.current += 1           
            return dat;
            
if __name__ == '__main__':
    for i, x in enumerate( BatchReader() ):
        print i, len(x), x