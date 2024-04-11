import numpy as np
class Kernel:
    # shape is a shapelike object where the nth element is in the format: (channels, rows, columns)
    def __init__(self, shape):
        if len(shape) != 3:
            raise Exception(f'Kernel dimension must be 3. Given shape: {shape}')
        if shape[1] != shape[2]:
            raise Exception(f'Kernel row count must equal column count. Given shape: {shape}')
        self.shape = shape
        self.weights = np.around(np.random.random(shape),1)

    # input is a 3d numpy array (channels, rows, columns)
    def convolve(self, input, stride):
        if input.shape[0] != self.shape[0]:
            raise Exception(f'Kernel channel count doesn\'t match Input channel count. Kernel shape: {self.shape}  Input shape: {input.shape}')

        if input.shape[1] < self.shape[1] or input.shape[2] < self.shape[2]:
            raise Exception(f'Input bigger than kernel. Kernel shape: {self.shape}  Input shape: {input.shape}')
        
        if input.shape[1] != input.shape[2]:
            raise Exception(f'Input row count must equal column count! Input shape: {input.shape}')
        
        output_size = int((input.shape[1] - self.weights.shape[1])/stride) + 1
        output = np.zeros((output_size, output_size))
        for i in range(output_size):
            for j in range(output_size):
                row_start = i*stride
                row_end = row_start + self.shape[1]
                col_start = j*stride
                col_end = col_start + self.shape[2]
                output[i,j] = np.sum(input[:,row_start:row_end, col_start:col_end] * self.weights)
        return output


                

            