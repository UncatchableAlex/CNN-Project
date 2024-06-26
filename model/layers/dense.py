# This is our fully-connected layer
from model.layers.layer import Layer # type: ignore[import-not-found] 
import numpy as np
from typing_extensions import override
from model.optimizers.adam import Adam # type: ignore[import-not-found] 

# this is a class for describing fully-connected hidden layers
class Dense(Layer):
    @override
    # input
    def __init__(self, input_shape, output_shape, activation='sigmoid', bias=0.01):
        
        self.activation_label = activation
        super().__init__(input_shape, output_shape, activation)
        # each column represents the weights leading into a node in the next layer
        # each row represents the weights from a node in the last layer
        self.w = np.random.uniform(-1, 1, (input_shape[0], output_shape[0]))
        #self.w = np.full((input_shape[0], output_shape[0]), 0.01)
        # add a bias term
        self.b = np.full(output_shape, bias)
        #self.w = w
        self.input_shape = input_shape
        self.output_shape = output_shape


        
    # input is a column vector with n inputs, one for each node in the layer
    # returns a column vector which should serve as input to the next layer
    @override
    def forward(self, input):
        """
        Performs the forward pass of the model.

        Args:
            input (tensor): The input tensor to the model.

        Returns:
            tensor: The output tensor of the model after applying the weights of dense layer and adding the bias.
        """
        self.input = input
        self.input_activ = self.activation[0](input)
        # apply the activation function and return the forward operation as input to the next layer (if this layer is hidden)
        return (self.w.T @ self.input_activ) + self.b
    
    def compile(self,optimizer):
        if optimizer == 'adam':
            self.w_optimizer = Adam()
            self.b_optimizer = Adam()
        else:
            raise Exception('Unknown optimizer. Please use Adam')

    # grad_output has 1 row and n columns where n is the number of nodes in the next layer
    # grad_output this is where we leave off for the night. we are concerned that grad_output is a 1d array that we are going to transpose and it will not end well
    @override
    def backward(self, grad_output):
        """
        Performs the backward pass of the dense layer.

        Args:
            ouput_grad (ndarray): The grad output from previous layer.

        Returns:
            ndarray: The blame factor 
        """
        # use the formula from Adam's book (formula 4.28, page 103, Machine Learning by Tom M Mitchell)
        self.blame = self.activation[1](self.input) * (self.w @ grad_output)
        # take a step in grad_output direction (these are the blames from the next layer)
        # this is exactly how adam does it in his notes. we are using a numpy trick to do a
        # (n,1) * (1,m) = (n,m) broadcast.
        self.w += self.w_optimizer.update(grad_output.T * self.input_activ)
        self.b += self.b_optimizer.update(grad_output)
        # blame is a column vector with an element for every node in this layer
        return self.blame
