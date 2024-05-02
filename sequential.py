from model.layers.dense import Dense
import os
from sklearn.model_selection import train_test_split# type: ignore
import glob
import logging
from PIL import Image# type: ignore
from model.layers.conv2d import Conv2D
from model.layers.maxpool2d import MaxPool2D
from model.layers.flatten import Flatten
from model.layers.output import Output
from model.layers.dropout import DenseDropout, ConvDropout
from kernel import Kernel
import numpy as np
import h5py # type: ignore

class Sequential:
    def __init__(self):
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)
    
    def forward(self, input, index=0):
        if index >= len(self.layers):
            return input
        output = self.layers[index].forward(input)
        return self.forward(output, index + 1)
    
    def backward(self, output, index=None):
        if index is None:
            index = len(self.layers) - 1
        if index < 0:
            return output
        grad = self.layers[index].backward(output)
        return self.backward(grad, index - 1)

    def compile(self,optimizer):
        for layer in self.layers:
            layer.compile(optimizer=optimizer)


    def preprocess_dataset(self, data_dir, map_dictionary):
        # Get all image file paths
        all_image_files = glob.glob(os.path.join(data_dir, '*.png'))
        if not all_image_files:
            logging.error("No image files found in the specified directory.")
            return None, None, None, None, None

        # Prepare data lists
        filenames = [os.path.basename(file) for file in all_image_files]
        classes = []
        for filename in filenames:
            for identifier, label_vector in map_dictionary.items():
                if identifier in filename:
                    classes.append(label_vector)
                    break
            else:
                logging.warning(f"No matching identifier found for file: {filename}")
                classes.append([0.0] * len(next(iter(map_dictionary.values()))))

        images = [np.array(Image.open(file).convert('RGB')) / 255.0 for file in all_image_files]
        # Normalize and ensure RGB
        images = [image.transpose(2, 0, 1) for image in images]

        # Split into train and validation sets
        train_images, validation_images, train_labels, validation_labels = train_test_split(
            images, classes, test_size=0.2, random_state=32
        )

        # Create indices array
        indices = np.arange(len(train_images))

        return train_images, train_labels, validation_images, validation_labels, indices

    def training(self, train_value, train_target, indices, number_of_iteration):
        number_of_data = len(train_value)
        for j in range(number_of_iteration):
            # Shuffle the indices array for each iteration
            np.random.shuffle(indices)
            for i in range(number_of_data):
                # Use the shuffled indices to access training data
                index = indices[i]
                input = train_value[index]
                target = train_target[index]
                # Change the final output
                self.layers[-1].target = target
                output = self.forward(input, index=0)
                print(output)
                self.backward(output=output)
        

    
    

    def save(self, file_name):
        f = h5py.File(file_name, 'a')
        keys = [f'layer_{i}' for i in range(len(self.layers))]
        types = [str(type(layer)).split('.')[-1].lower()[:-2] for layer in self.layers]
        f.attrs['layer_types'] = types
        f.attrs['layer_keys'] = keys
        for i in range(len(self.layers)):
            layer_key = keys[i]
            layer_type = types[i]
            layer_obj = self.layers[i]
            f.create_group(layer_key)
            layer = f[layer_key]
            layer.attrs['input_shape'] = layer_obj.input_shape
            layer.attrs['output_shape'] = layer_obj.output_shape        
            if layer_type == 'conv2d':
                layer.attrs['filters'] = len(layer_obj.kernels)
                layer.attrs['kernel_size'] = layer_obj.kernel_size
                layer.attrs['stride'] = layer_obj.stride
                layer.attrs['activation'] = layer_obj.activation_label
                layer.create_dataset('bias', data=layer_obj.bias)
                layer.attrs['kernels'] = [f'kernel_{i}' for i in range(len(layer_obj.kernels))]
                for kern_key, kern_obj in zip(np.array(layer.attrs['kernels']), layer_obj.kernels):
                    layer.create_group(kern_key)
                    layer[kern_key].create_dataset('weights', data=kern_obj.weights)
            elif layer_type == 'maxpool2d':
                layer.attrs['pool_size'] = layer_obj.pool_size
                layer.attrs['stride'] = layer_obj.stride
            elif layer_type == 'output':
                layer.attrs['target_vec'] = layer_obj.target_vec
                layer.attrs['activation'] = layer_obj.activation_label
                #loss_func = layer.attrs['loss_func'] = layer_obj.loss_func
            elif layer_type == 'input':
                pass
            elif layer_type == 'flatten':
                pass
            elif layer_type == 'dense':
                layer.attrs['activation'] = layer_obj.activation_label
                layer.create_dataset('bias', data=layer_obj.b)
                layer.create_dataset('weights', data=layer_obj.w)
            elif layer_type == 'conv_dropout':
                layer.attrs['dropout_rate'] = layer_obj.dropout_rate
            elif layer_type == 'dense_dropout':
                layer.attrs['dropout_rate'] = layer_obj.dropout_rate
            else:
                raise Exception(f'unrecognized layer: {layer_type}')            
        f.close()




    @staticmethod
    def load(file_name):
        f = h5py.File(file_name, 'r')
        seq = Sequential()
        layer_types = np.array(f.attrs['layer_types'])
        layer_keys = np.array(f.attrs['layer_keys'])
        for layer_type, layer_key in zip(layer_types, layer_keys):
            layer_obj = None
            layer = f[layer_key]
            input_shape = tuple(layer.attrs['input_shape'])
            output_shape = tuple(layer.attrs['output_shape'])
            if layer_type == 'conv2d':
                filters = layer.attrs['filters']
                kernel_size = layer.attrs['kernel_size']
                stride = layer.attrs['stride']
                activation = layer.attrs['activation']
                layer_obj = Conv2D(input_shape, output_shape, filters, kernel_size, stride, activation)
                layer_obj.bias = np.array(layer['bias'])
                kernels = []
                for kern in np.array(layer.attrs['kernels']):
                    weights = np.array(layer[kern]['weights'])
                    new_kernel_obj = Kernel(shape=(input_shape[0], kernel_size, kernel_size), stride=stride)
                    new_kernel_obj.weights = weights
                    kernels.append(new_kernel_obj)
                layer_obj.kernels = kernels
            elif layer_type == 'maxpool2d':
                pool_size = layer.attrs['pool_size']
                stride = layer.attrs['stride']
                layer_obj = MaxPool2D(input_shape, output_shape, pool_size, stride)
            elif layer_type == 'output':
                target_vec = layer.attrs['target_vec']
                activation = layer.attrs['activation']
                #loss_func = layer.attrs['loss_func']
                layer_obj = Output(input_shape, output_shape, target_vec, activation)
            elif layer_type == 'input':
                pass
            elif layer_type == 'flatten':
                layer_obj = Flatten(input_shape, output_shape)
            elif layer_type == 'dense':
                activation = layer.attrs['activation']
                bias = np.array(layer['bias'])
                weights = np.array(layer['weights'])
                layer_obj = Dense(input_shape, output_shape, activation)
                layer_obj.b = bias
                layer_obj.w = weights
            elif layer_type == 'conv_dropout':
                dropout_rate = layer.attrs['dropout_rate']
                layer_obj = ConvDropout(input_shape, output_shape, dropout_rate)
            elif layer_type == 'dense_dropout':
                dropout_rate = layer.attrs['dropout_rate']
                layer_obj = DenseDropout(input_shape, output_shape, dropout_rate)
            else:
                raise Exception(f'unrecognized layer: {layer_type}')
            seq.add(layer_obj)
        f.close()
        return seq
    


