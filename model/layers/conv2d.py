from model.layers.layer import Layer # type: ignore[import-not-found] 
from model.optimizers.adam import Adam
from kernel import Kernel # type: ignore[import-not-found] 
from typing_extensions import override
import numpy as np

class Conv2D(Layer):
    @override
    def __init__(self, input_shape, output_shape, filters, kernel_size, stride=1, activation='sigmoid', bias=0.01):
        self.activation_label = activation
        super().__init__(input_shape, output_shape, activation)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.kernels = [Kernel(shape=(input_shape[0], kernel_size, kernel_size), stride=stride) for _ in range(filters)]
        self.blame = None
        self.bias = np.full(output_shape, bias)
        self.output = np.zeros(output_shape)
    
    def compile(self,optimizer):
        """
        Compiles the model with the specified optimizer.

        Args:
            optimizer (str): The type of optimizer to use. Currently, only 'adam' is supported.

        Raises:
            Exception: If an unknown optimizer is provided.
        """
        if optimizer == 'adam':
            self.b_optimizer = Adam()
        else:
            raise Exception('Unknown optimizer. Please use Adam')
        for k in self.kernels:
            k.compile(optimizer)

    @override
    def forward(self, input):
        """
        Performs the forward pass of the model.

        Args:
            input (tensor): The input tensor to the model.

        Returns:
            tensor: The output tensor of the model after applying the convolutions and adding the bias.
        """
        self.input = input
        self.input_activ = self.activation[0](input)
        # convolve each kernel with result of the activation function and the input
        for i,kern in enumerate(self.kernels):
            self.output[i] = kern.forward_convolve(self.input_activ)
        return self.output + self.bias


    @override
    # output_grads is a 4d tensor
    # https://www.youtube.com/watch?v=Lakz2MoHy6o&t=1292s
    # In the video, his variables are:
    #  x -> self.input
    #  y -> output_grads  (a 2d matrix the same size as our forward prop output)
    #  k-> self.kernels
    def backward(self, output_grads):
        """
        Performs the backward pass of the conv2d layer.

        Args:
            ouput_grad (ndarray): The grad output from previous layer.

        Returns:
            ndarray: The blame factor 
        """
        self.blame = np.zeros_like(self.input) if self.blame is None else self.blame
        for j in range(self.blame.shape[0]):
            self.blame[j] = np.sum(np.array([kern.channel_blame(j, output_grad) for kern, output_grad in zip(self.kernels, output_grads)]), axis=0)
        
        # for each kernel and it's associated output gradient, update the weights
        for kern, output_grad in zip(self.kernels, output_grads):
            kern.update_weights(self.input_activ, output_grad)

        # update biases:
        self.bias += self.b_optimizer.update(output_grads)
        #print(np.max(np.abs(self.bias)))
        
        return self.activation[1](self.input) * self.blame

    