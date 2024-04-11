from kernel import Kernel
import numpy as np

kern = Kernel((2, 2))
print(kern.weights)
input_matrix = np.array([
        [1,2],
        [4,5]
])
print(kern.weights[0:2, 0:2] * input_matrix)
print(np.sum(input_matrix*kern.weights))
