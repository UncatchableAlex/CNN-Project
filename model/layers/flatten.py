from model.layers.layer import Layer # type: ignore[import-not-found]
from typing_extensions import override
import numpy as np


class Flatten(Layer):
    @override
    def __init__(self, input_shape, output_shape, activation):
        super().__init__(input_shape, output_shape, activation)

    @override
    def forward(self, input):
        # return our input as a column vector
        return input.reshape(-1,1)
    
    @override
    def backward(self, grad_output):
        # return our column vector grad_output as a matrix:
        return grad_output.reshape(self.input_shape)
    
    def compile(self, optimizer):
        pass