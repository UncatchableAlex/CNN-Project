from abc import ABC, abstractmethod
import numpy as np

class Layer(ABC):
    @abstractmethod
    def __init__(self, input_shape, output_shape, activation):
        self.input_shape = input_shape
        self.output_shape = output_shape
        # a dictionary with the string name of each activation function as the key and a 2 element list of lambda 
        # activation functions. The first function in the list is the activation function and the second element is the derivative
        # TODO add real activation functions
        sig = lambda x: 1/(1+np.exp(-x))
        activation_funcs = {
            'relu': [lambda x: x, lambda x: 1],
            'sigmoid': [sig, lambda x:  sig(x)*(1-sig(x))]
        }
        self.activation = activation_funcs[activation]


    @abstractmethod
    def forward(self, input):
        pass

    @abstractmethod
    def backward(self, grad_output):
        pass

