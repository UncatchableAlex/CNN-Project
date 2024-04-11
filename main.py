from kernel import Kernel
import numpy as np

kern = Kernel((2, 2, 2))
print(kern.weights)
input_matrix = np.array(
    [
        [[1, 1, 4, 5], [2, 2, 8, 9], [9, 8, 7, 6], [5, 4, 3, 2]],
        [[3, 3, 3, 5], [4, 4, 2, 4], [2, 4, 5, 6], [1, 4, 5, 6]],
    ]
)
#print(kern.weights[0:2, 0:2] * input_matrix)
#print(np.sum(input_matrix * kern.weights))

print(kern.convolve(input=input_matrix, stride=2))
