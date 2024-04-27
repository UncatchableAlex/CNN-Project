from model.layers.layer import Layer # type: ignore[import-not-found] 
from typing_extensions import override
import numpy as np

class ConvDropout(Layer):
    @override
    def __init__(self, input_shape, output_shape, dropout_rate=0.5):
        super().__init__(input_shape, output_shape, '')
        if input_shape != output_shape:
            raise Exception('Input shape must match output shape')
        
        #self.positions = None
        self.dropout_rate = dropout_rate
        self.dropout_num = int(self.input_shape[0] * self.input_shape[1] * self.input_shape[2] * self.dropout_rate)

    
    def compile(self,optimizer):
        pass

    @override
    def forward(self, input):
        z_coord = np.random.randint(low=0, high=self.input_shape[0], size=self.dropout_num)
        x_coord = np.random.randint(low=0, high=self.input_shape[1], size=self.dropout_num)
        y_coord = np.random.randint(low=0, high=self.input_shape[2], size=self.dropout_num)
       # self.positions = np.stack((z_coord, x_coord, y_coord))
        input_copy = np.copy(input)
        input_copy[z_coord,x_coord,y_coord] = 0.0
        return input_copy


    @override
    def backward(self, output_grads):
        return output_grads
        # this is a matrix of the channel coordinates for each of the max values that we outputed upon forward prop
        channel_pos = self.positions[0]
        # and a matrix for the row coordinates
        row_pos = self.positions[1]
        # and a matrix for the col coordinates
        col_pos = self.positions[2]
        print(output_grads[channel_pos, row_pos, col_pos])
        output_grads[channel_pos, row_pos, col_pos] = 0.0
        return output_grads
    



class DenseDropout(Layer):
    @override
    # input
    def __init__(self, input_shape, output_shape, dropout_rate=0.5):
        super().__init__(input_shape, output_shape, '')

        if input_shape != output_shape:
            raise Exception('Input shape must match output shape')
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.dropout_rate = dropout_rate
        self.dropout_num = int(self.input_shape[0] * self.dropout_rate)
       # self.positions = None

        
    # input is a column vector with n inputs, one for each node in the layer
    # returns a column vector which should serve as input to the next layer
    @override
    def forward(self, input):
        x_coord = np.random.randint(low=0, high=self.input_shape[0], size=self.dropout_num)
       # self.positions = x_coord
        input_copy = np.copy(input)
        input_copy[x_coord, 0] = 0.0
        return input_copy

    # grad_output has 1 row and n columns where n is the number of nodes in the next layer
    # grad_output this is where we leave off for the night. we are concerned that grad_output is a 1d array that we are going to transpose and it will not end well
    @override
    def backward(self, output_grads):
        return output_grads
      #  print(output_grads[self.positions,0])
        # output_grads[self.positions, 0] = 0.0
        # return output_grads

