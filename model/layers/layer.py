from abc import ABC, abstractmethod
import numpy as np

class Layer(ABC):
    @abstractmethod
    def __init__(self, input_shape, output_shape, activation):
        """
        Initializes the layer with the given input shape, output shape, and activation function.

        Args:
            input_shape (tuple): The shape of the input tensor.
            output_shape (tuple): The shape of the output tensor.
            activation (str): The name of the activation function to be used.
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_shape = input_shape
        self.output_shape = output_shape
        # a dictionary with the string name of each activation function as the key and a 2 element list of lambda 
        # activation functions. The first function in the list is the activation function and the second element is the derivative
        sig = lambda x: 1/(1+np.exp(-x))
        activation_funcs = {
            'relu': [lambda x: np.maximum(x, 0.0), lambda x: np.where(x > 0, 1.0, 0.0)],
            'sigmoid': [sig, lambda x:  sig(x)*(1-sig(x))],
            '': None
        }
        self.activation = activation_funcs[activation]


    @abstractmethod
    def forward(self, input):
        pass

    @abstractmethod
    def backward(self, grad_output):
        pass

