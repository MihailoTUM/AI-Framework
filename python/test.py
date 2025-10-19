from Tensor import Tensor
import numpy as np
import sys
sys.path.append(r"C:\Users\drini\OneDrive\Desktop\cuda\CUDA\bin")
import mikideeplib as mk

input_array = mk.TensorFloat(1, 8, "CPU", np.random.rand(1, 8).astype(np.float32))
input = Tensor(1, 8, "CPU", input_array)

weights1 = Tensor(8, 6, "CPU")
bias1 = Tensor(1, 6, "CPU")

weights2 = Tensor(6, 2, "CPU")
bias2 = Tensor(1, 2, "CPU")

out = (input @ weights1 + bias1) @ weights2 + bias2
graph = out.backward()
print(graph)


