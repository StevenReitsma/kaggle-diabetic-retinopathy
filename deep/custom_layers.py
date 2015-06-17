from lasagne.layers import Layer, MergeLayer
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

class BatchConcatLayer(Layer):

	def __init__(self, incoming, nr_views, **kwargs):
		super(BatchConcatLayer, self).__init__(incoming, **kwargs)

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
		
	
class ConcatLayer(MergeLayer):
    """
    Concatenates multiple inputs along the specified axis. Inputs should have
    the same shape except for the dimension specified in axis, which can have
    different sizes.

    Parameters
    -----------
    incomings : a list of :class:`Layer` instances or tuples
        The layers feeding into this layer, or expected input shapes

    axis : int
        Axis which inputs are joined over

    See Also
    ---------
    concat : Shortcut

    """
    def __init__(self, incomings, axis=1, **kwargs):
        super(ConcatLayer, self).__init__(incomings, **kwargs)
        self.axis = axis

    def get_output_shape_for(self, input_shapes):
	        
	sizes = [input_shape[self.axis] for input_shape in input_shapes]
	
        output_shape = list(input_shapes[0])  # make a mutable copy
	if self.axis != 0:
        	output_shape[self.axis] = sum(sizes)
	
        return tuple(output_shape)

    def get_output_for(self, inputs, **kwargs):      
	return T.concatenate(inputs, axis=self.axis)
