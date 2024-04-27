from model.layers.layer import Layer # type: ignore[import-not-found] 
from typing_extensions import override
import numpy as np

class MaxPool2D(Layer):
    @override
    def __init__(self, input_shape, output_shape, pool_size, stride=1):
        super().__init__(input_shape, output_shape, '')
        if (pool_size[0] != pool_size[1]):
            raise Exception(f'Only square pool sizes are allowed. Entered pool_size: {pool_size}')
        if (len(pool_size) != 2):
            raise Exception(f'pool size must be 2d. Entered pool_size: {pool_size}')
        
        # notify the user if they messed up their output size 
        expected_output_size = int((input_shape[1] - pool_size[0])/stride) + 1
        if expected_output_size != output_shape[1]:
            raise Exception(f'erroneious output shape. Found {output_shape} expected ({expected_output_size}, {expected_output_size})')

        
        self.stride = stride
        self.pool_size = pool_size
        self.blames = np.zeros(self.input_shape)
        self.output = np.zeros(output_shape)
        self.positions = np.zeros(self.output.shape, dtype=(int, 3))
    
    def compile(self,optimizer):
        pass

    @override
    def forward(self, input):
        for i in range(self.output_shape[1]): # rows
            for j in range(self.output_shape[2]): # columns
                row_start = i*self.stride
                row_end = row_start + self.pool_size[0]
                col_start = j*self.stride
                col_end = col_start + self.pool_size[1]
                for k in range(input.shape[0]): # channels
                    frame = input[k,row_start:row_end, col_start:col_end]
                    self.output[k,i,j] = np.max(frame)
                    # get the coordinates in the frame with the max value
                    self.positions[k,i,j,1:] = np.unravel_index(np.argmax(frame), frame.shape)
                    # add channel to positions
                    self.positions[k,i,j,0] = k
                    # add offsets to get the coordinates in the output matrix with max value of this frame
                    self.positions[k,i,j,1] += row_start
                    self.positions[k,i,j,2] += col_start
        return self.output


    @override
    def backward(self, output_grads):
        # this is a matrix of the channel coordinates for each of the max values that we outputed upon forward prop
        channel_pos = self.positions[:, :, :, 0]
        # and a matrix for the row coordinates
        row_pos = self.positions[:, :, :, 1]
        # and a matrix for the col coordinates
        col_pos = self.positions[:, :, :, 2]
        self.blames[channel_pos, row_pos, col_pos] = output_grads
        return self.blames

