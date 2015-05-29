# -*- coding: utf-8 -*-

import batchreader as br
import numpy as np

class RandomBatchReader (br.BatchReader):
    
    """
    Reads chunks of data, with sub-chunks randomly.. 
    The sub-chunk size is set with argument random_size
    
    If this is set to the amount of patches per image, it will read
    exactly the patches belonging to images randomly.
    
    """
    
    
    def __init__(self, filepath="../data/preprocessed.h5", batchsize=72900, dataset="data", random_size=729):
        if not batchsize % random_size == 0:
            raise Exception('batchsize must be n times random_size')

        # Super constructor
        br.BatchReader.__init__(self, filepath, batchsize, dataset)
        
        self.random_size = random_size
        
        # Indices to read the random batches from.
        self.start_indices = np.arange(self.dimensions[0]/self.random_size) * self.random_size 
        
        np.random.shuffle(self.start_indices)
        
        # Cache indices per batch to write to
        # For a batch size of 100, with a random_size of 10
        # this would be 0, 10, 20, ..., 80, 90
        self.write_indices = np.arange(self.batch_size/self.random_size) * self.random_size
        
    def next(self):
        
        if self.current >= len(self.start_indices):
                self.file.close()
                raise StopIteration  
        else:
        
            dat = np.zeros( (self.batch_size, self.dimensions[1]), dtype=np.float32)            
            
            for index in self.write_indices:
                
                #Last batch is likely smaller, take the remainder
                if self.current >= len(self.start_indices):
                    dat = dat[:index]
                    break
                
                startIndex = self.start_indices[self.current]
                endIndex = startIndex + self.random_size
                    
                dat[index:index+self.random_size] = self.dset[ startIndex:endIndex  ]

                self.current += 1;
     
            return dat;
    
if __name__ == "__main__":
    tot = 0
    for chunk_of_data in RandomBatchReader():
        print chunk_of_data
        tot += len(chunk_of_data)
        
    print tot
        