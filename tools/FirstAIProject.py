import numpy as np
import math
# import torch as nn

arr = np.array([1,2,3])
arr1 = np.array([2,3,4])

mlx = np.array([[1,2,3], [2,3,4],[3,4,5]])

# mlx.shape(3, 3)

mlx.shape

# print(mlx * arr)

# print(mlx[0])

# print(mlx[1][2])


for row in mlx :
    for s in row:
        print(s)

smlx = mlx.flatten()

print(smlx)

print(mlx > 3)