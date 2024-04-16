import numpy as np
from layer import Layer

class Loss(Layer):
    def __init__(self, input_shape, output_shape, target_vec, activation, loss_func):
        super().__init__(input_shape, output_shape, activation)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.loss_func = lambda x: 0.5*(self.activation[0](x) - target_vec)**2
        self.loss_grad = lambda x: self.activation[1](x)*(self.activation[0](x) - target_vec)

    def forward(self, input):
        return self.loss_func(input)
        

    def backward(self, grad_output):
        return self.loss_grad(grad_output)

