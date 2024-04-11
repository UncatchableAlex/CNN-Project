import numpy as np
class Kernel:
    # shape is a shapelike object where the nth element is in the format: (channels, rows, columns)
    def __init__(self, shape):
        if len(shape) != 3:
            raise Exception(f'Kernel dimension must be 3. Given shape: {shape}')
        if shape[1] != shape[2]:
            raise Exception(f'Row count must equal column count. Given shape: {shape}')
        self.shape = shape
        self.weights = np.random.random(shape)

    # input is a 3d numpy array (channels, rows, columns)
    def apply(self, input, stride):
        if input.shape[0] != self.shape[0]:
            raise Exception(f'Kernel channel count doesn\'t match Input channel count. Kernel shape: {self.shape}  Input shape: {input.shape}')

        if input.shape[1] < self.shape[1] or input.shape[2] < self.shape[2]:
            raise Exception(f'Input bigger than kernel. Kernel shape: {self.shape}  Input shape: {input.shape}')
        
        
        for i in range(0, 1 + input.shape[1] - self.weights.shape[1], self.stride):
            for j in range(0, 1 + input.shape[2]- self.weights.shape[1], self.stride):
                

            