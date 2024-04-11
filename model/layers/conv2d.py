from layer import Layer
from kernel import Kernel
from typing_extensions import override
import numpy as np

class Conv2D(Layer):
    @override
    def __init__(self, input_shape, output_shape, filters, kernel_size, stride=1, activation='relu', bias=0.0):
        super().__init__(input_shape, output_shape)

        # a dictionary with the string name of each activation function as the key and a 2 element list of lambda 
        # activation functions. The first function in the list is the activation function and the second element is the derivative
        # TODO add real activation functions
        activation_funcs = {
            'relu': [lambda x: x, lambda x: 1]
        }

        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation_funcs[activation]
        self.bias = bias
        self.kernels = [Kernel((input_shape[0], kernel_size, kernel_size)) for _ in range(filters)]

    

    @override
    def forward(input):
        pass


    @override
    def backward(input):
        pass
