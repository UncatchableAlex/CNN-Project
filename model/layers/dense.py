# This is our fully-connected layer
from layers import Layer
import numpy as np
from typing_extensions import override

class Dense(Layer):
    @override
    # input
    def __init__(self, input_shape, output_shape, nodes, w, activation='sigmoid', bias=0.0):
        super().__init__(input_shape, output_shape, activation)
        # each column represents the weights leading into a node in the next layer
        # each row represents the weights from a node in the last layer
        #self.w = np.random.uniform(-0.5, 0.5, (nodes, output_shape[0]))
        self.w = w
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_blame = np.zeros(self.input_shape[0])

        
    @override
    def forward(self, input):
        self.input = input
        self.output =  self.activation[0](input @ self.w)
        return self.output

    @override
    # grad_output has 1 row and n columns where n is the number of nodes in the next layer
    # grad_output this is where we leave off for the night. we are concerned that grad_output is a 1d array that we are going to transpose and it will not end well
    # 
    def backward(self, grad_output):
        self.w += 1.0 * grad_output
        # just trust us...
        blame = ((self.w @ (self.activation[1](grad_output).T)) * self.output.T).T
        return blame
