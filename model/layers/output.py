import numpy as np
from model.layers.layer import Layer # type: ignore[import-not-found] 

class Output(Layer):
    """
        Initializes the Output layer.

        Args:
            input_shape (tuple): The shape of the input tensor.
            output_shape (tuple): The shape of the output tensor.
            target_vec (ndarray): The target vector for training.
            activation (str): The activation function to be used.
    """
    def __init__(self, input_shape, output_shape, target_vec, activation):
        self.activation_label = activation
        super().__init__(input_shape, output_shape, activation)
        self.target = target_vec
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.loss_func = lambda x: 0.5*(self.activation[0](x) - self.target)**2
        self.loss_grad = lambda x: self.activation[1](x)*(self.activation[0](x) - self.target)
        self.loss_func = lambda x: x * (1-x) *(self.target - x)

    def forward(self, input):
        """
        Performs the forward pass of the Output layer.

        Args:
            input (ndarray): The input tensor.

        Returns:
            ndarray: The output tensor after applying the activation function.
        """
        return self.activation[0](input)
        

    
    def backward(self, model_output):
        """
        Performs the backward pass of the Output layer.

        Args:
            ouput (ndarray): The model output.

        Returns:
            ndarray: The blame factor
        """
        self.blame = self.loss_func(model_output)
        return self.blame
    

    def compile(self,optimizer):
        pass