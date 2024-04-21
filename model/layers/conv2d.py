from model.layers.layer import Layer # type: ignore[import-not-found] 
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
        self.kernels = [Kernel(shape=(input_shape[0], kernel_size, kernel_size), stride=stride) for _ in range(filters)]
    
    
    def compile(self,optimizer):
        for k in self.kernels:
            k.compile(optimizer)

    @override
    def forward(self, input):
        self.input = input
        self.input_activ = self.activation[0](input)
        # convolve each kernel with result of the activation function and the input
        return np.array([k.forward_convolve(self.input_activ) for k in self.kernels])


    @override
    # output_grads is a 4d tensor
    # https://www.youtube.com/watch?v=Lakz2MoHy6o&t=1292s
    # In the video, his variables are:
    #  x -> self.input
    #  y -> output_grads 
    #  k-> self.kernels
    def backward(self, output_grads):
        return [self.activation[1](self.input) * kern.backward_convolve(self.input, output_grad) for kern, output_grad in zip(self.kernels, output_grads)]
        #
        # This sick one-liner is equivalent to the following imerative code:
        #
        # outputs = []
        # for i in range(len(output_grads)):
        #     output_grad = output_grads[i]
        #     kern = self.kernels[i]
        #     output = kern.backward_convolve(self.input, output_grad)
        #     outputs.append(output)

