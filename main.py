from model.layers.dense import Dense
from final_project.model.layers.output import Loss
import numpy as np


layer1 = Dense((2,0), (2,0), 2, np.array([[0.1,0.4], [0.8,0.6]]))
layer2 = Dense((2,0), (1,0), 2, np.array([[0.3], [0.9]]))
layer3 = Loss((1,0), (1,0), 1, '', '')

input = np.array([[1,2]]).T
layer3.forward(layer2.forward(layer1.forward(input)))


