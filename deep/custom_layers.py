from lasagne.layers import Layer
import numpy as np
import theano.tensor as T
from params import *

class FlattenLayer(Layer):

    def __init__(self, incoming, **kwargs):
        super(FlattenLayer, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], 1, 1, input_shape[1]*input_shape[2]*input_shape[3])

    def get_output_for(self, input, **kwargs):
        return input.reshape((input.shape[0], 1, 1, -1))

class MergeLayer(Layer):

	def __init__(self, incoming, nr_views, **kwargs):
		super(MergeLayer, self).__init__(incoming, **kwargs)

		self.nr_views = nr_views

	def get_output_shape_for(self, input_shape):							
		#if input_shape[0] != None:
		#	return (64,input_shape[1]*self.nr_views,input_shape[2],input_shape[3])		
		#else:
		return (None,input_shape[1]*self.nr_views,input_shape[2],input_shape[3])

	def get_output_for(self, input, *args, **kwargs):				
		ninput = input.reshape((2, input.shape[0] / 2, input.shape[1], input.shape[2], input.shape[3]))
		ninput = ninput.transpose(1, 2, 0, 3, 4)
		ninput = ninput.reshape((input.shape[0] / 2, input.shape[1] * 2, input.shape[2], input.shape[3]))

		return ninput
		
	
	
