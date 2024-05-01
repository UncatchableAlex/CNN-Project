from model.layers.dense import Dense
import os
import sys
from sklearn.model_selection import train_test_split
import glob
import logging
import pandas as pd
from PIL import Image
import numpy as np


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

    def save(self):
        pass
            
    def preprocess_dataset(self,data_dir, identifier):
        # Get all image file paths
        all_image_files = glob.glob(os.path.join(data_dir, '*.jpg'))
        if not all_image_files:
            logging.error("No image files found in the specified directory.")
            return None, None, None, None

        # Prepare data lists
        filenames = [os.path.basename(file) for file in all_image_files]
        classes = [1 if identifier in filename else 0 for filename in filenames]
        images = [np.array(Image.open(file).convert('RGB')) / 255.0 for file in all_image_files]  # Normalize and ensure RGB

        # Split into train and validation sets
        train_images, validation_images, train_labels, validation_labels = train_test_split(
            images, classes, test_size=0.2, random_state=32
        )

        return train_images, train_labels, validation_images, validation_labels

    def training(self,train_value,train_target,number_of_iteration):
        number_of_data =len(train_value)
        for j in range(number_of_iteration):
            for i in range(number_of_data):
                input=train_value[i]
                
                target=train_target[i]
                #change the final output
                self.layers[-1].target =target
                output =self.forward(input, index=0)
                print(output)
                self.backward(output=output)
            
            
            
            
    