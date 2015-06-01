from lasagne.layers import Layer

class FlattenLayer(Layer):

    def __init__(self, incoming, **kwargs):
        super(FlattenLayer, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], 1, 1, input_shape[1]*input_shape[2]*input_shape[3])

    def get_output_for(self, input, **kwargs):
        return input.reshape((input.shape[0], 1, 1, -1))
