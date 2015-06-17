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
		if input_shape[0] != None:
			return (input_shape[0]/self.nr_views,input_shape[1]*self.nr_views,input_shape[2],input_shape[3])		
		else:
			return (None,input_shape[1]*self.nr_views,input_shape[2],input_shape[3])

	def get_output_for(self, input, *args, **kwargs):		
		output_shape = self.get_output_shape_for(self.input_shape)
		if output_shape[0] != None:		
			return input.reshape(output_shape)
		else:
			return input
"""		new_input = T.btensor4()
#		for sample in xrange(0,output_shape[0]):	
#			new_input[sample] = T.concatenate((input[sample:],input[sample+output_shape#[0]:]),axis=1)	
	
		

class MergeTwoLayers(MergeLayer):
	def __init__(self, incomings, **kwargs):
		super(MergeTwoLayers,self).__init__(incomings,**kwargs)		
		if not isinstance(incomings, (list, tuple)):
			incomings = [incomings]
	
	def get_output_shape_for(self,input_shape):
		print input_shape		
		return input_shape
	
	
	def get_output_for(self, input, *args, **kwargs):
		parts = []		
		for layer in input:
			parts.extend(layer)
		return T.concatenate(parts,axis=0)
"""			
	
	
