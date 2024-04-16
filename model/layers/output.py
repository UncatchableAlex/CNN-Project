import numpy as np
from model.layers.layer import Layer # type: ignore[import-not-found] 

class Output(Layer):
    def __init__(self, input_shape, output_shape, target_vec, activation, loss_func):
        super().__init__(input_shape, output_shape, activation)
        self.input_shape = input_shape
        self.output_shape = output_shape
        #self.loss_func = lambda x: 0.5*(self.activation[0](x) - target_vec)**2
        #self.loss_grad = lambda x: self.activation[1](x)*(self.activation[0](x) - target_vec)
        self.loss_func = lambda x: x * (1-x) *(target_vec - x)

    def forward(self, input):
        return self.activation[0](input)
        

    #
    def backward(self, model_output):
        self.blame = self.loss_func(model_output)
        return self.blame

