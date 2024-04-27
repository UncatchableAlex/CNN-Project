import numpy as np
from model.optimizers.adam import Adam

class Kernel:
    # shape is a shapelike object where the nth element is in the format: (channels, rows, columns)
    def __init__(self, shape, stride):
        if len(shape) != 3:
            raise Exception(f'Kernel dimension must be 3. Given shape: {shape}')
        if shape[1] != shape[2]:
            raise Exception(f'Kernel row count must equal column count. Given shape: {shape}')
        if stride < 0:
            raise Exception(f'Stride cannot be negative. Given stride {stride}')
        self.shape = shape
        self.stride = stride
      #  self.weights = np.random.uniform(low=-1, high=1, size=shape)
        self.weights = np.full(shape, 0.01)
    

    def compile(self,w_optimizer):
        if w_optimizer == 'adam':
            self.w_optimizer = Adam()
        else:
            raise Exception('Unknown w_optimizer. Please use Adam')    
            

    def forward_convolve(self, input_activ):
        return Kernel._convolve(input_activ, self.weights, self.stride)
    

    def channel_blame(self, channel, grad_output):
        padded_grad_output = np.pad(grad_output, pad_width=self.shape[1] - 1)[np.newaxis, :, :]
        channel_weight_rot = np.rot90(self.weights[channel], k=2)[np.newaxis, :, :]
        return Kernel._convolve(padded_grad_output, channel_weight_rot, stride=self.stride)

    
    # https://www.youtube.com/watch?v=Lakz2MoHy6o&t=1349s
    def update_weights(self, input_activ, output_grad):
        weights_grad = np.array([Kernel._convolve(input_activ_2d[np.newaxis,:,:], output_grad[np.newaxis,:,:], stride=self.stride) for input_activ_2d in input_activ])
        # update weights in kernel. 
        self.weights += self.w_optimizer.update(weights_grad)
        
    # input is a 3d numpy array (channels, rows, columns)
    # output is a 2d numpy array (rows, columns)
    @staticmethod
    def _convolve(input, weights, stride):
        if input.shape[0] != weights.shape[0]:
            raise Exception(f'Weight channel count doesn\'t match Input channel count. Weight shape: {weights.shape}  Input shape: {input.shape}')

        if input.shape[1] < weights.shape[1] or input.shape[2] < weights.shape[2]:
            raise Exception(f'Input bigger than weight. Weight shape: {weights.shape}  Input shape: {input.shape}')
        
        if input.shape[1] != input.shape[2]:
            raise Exception(f'Input row count must equal column count! Input shape: {input.shape}')
        
        output_size = int((input.shape[1] - weights.shape[1])/stride) + 1
        output = np.zeros((output_size, output_size))
        for i in range(output_size):
            for j in range(output_size):
                row_start = i*stride
                row_end = row_start + weights.shape[1]
                col_start = j*stride
                col_end = col_start + weights.shape[2]
                output[i,j] = np.sum(input[:,row_start:row_end, col_start:col_end] * weights)
        return output

                

            