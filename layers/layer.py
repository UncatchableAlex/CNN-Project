from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    @abstractmethod
    def forward(input):
        pass

    @abstractmethod
    def backward(grad_output):
        pass

