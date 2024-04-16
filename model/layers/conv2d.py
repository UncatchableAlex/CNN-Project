from layer import Layer
from kernel import Kernel # type: ignore[import-not-found] 
from typing_extensions import override
import numpy as np

class Conv2D(Layer):
    @override
    def __init__(self, input_shape, output_shape, filters, kernel_size, stride=1, activation='relu', bias=0.0):
        super().__init__(input_shape, output_shape, activation)


        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.kernels = [Kernel((input_shape[0], kernel_size, kernel_size)) for _ in range(filters)]

    

    @override
    def forward(self, input):
        pass


    @override
    def backward(self, input):
        pass
