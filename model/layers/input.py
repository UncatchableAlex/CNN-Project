# This is our fully-connected layer
from model.layers.layer import Layer # type: ignore[import-not-found] 
import numpy as np
from typing_extensions import override

class Input(Layer):
    @override
    # input
    def __init__(self, input_shape, output_shape, nodes, w, activation='sigmoid', bias=0.0):
        super().__init__(input_shape, output_shape, activation)
        # each column represents the weights leading into a node in the next layer
        # each row represents the weights from a node in the last layer
        #self.w = np.random.uniform(-0.5, 0.5, (nodes, output_shape[0]))
        self.w = w
        # self.input_shape = input_shape
        # self.output_shape = output_shape
        # self.input_blame = np.zeros(self.input_shape[0])

        
    # input is a column vector with n inputs, one for each node in the layer
    # returns a column vector which should serve as input to the next layer
    @override
    def forward(self, input):
        self.input = input
        # apply the activation function and return the forward operation as input to the next layer (if this layer is hidden)
        return self.w.T @ input

    # grad_output has n row and 1 columns where n is the number of nodes in the next layer
    # Nothing needs to be done for backpropagation for the input layer. It just has to update the weights
    @override
    def backward(self, grad_output):
        self.w += 1.0 * grad_output * self.activation[0](self.input)
