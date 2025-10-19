# import sys
# # sys.path.append(r"C:\Users\drini\OneDrive\Desktop\cuda\CUDA")
import mikideeplib as mk
from Tensor import Tensor
import numpy as np

# tensor_1 = mk.TensorFloat(3, 3, "CPU")
# tensor_1.print()
# print("\n")

# tensor_2 = mk.TensorFloat(3, 3, "CPU")
# tensor_2.print()
# print("\n")

# result = tensor_1 + tensor_2
# result.print()

# print("\n")

# matmul = tensor_1 @ tensor_2
# matmul.print()

tensor = mk.TensorFloat(3, 3, "CPU")
v = tensor.getMatrix()
print(v)

print("\n")

tensor_1 = mk.TensorFloat(3, 3, "CPU", np.random.rand(3, 3).astype(np.float32))
v1 = tensor_1.getMatrix()
print(v1)

print("\n")

result = tensor + tensor_1
print(result.getMatrix())

print("\n")

t = result.transpose()
print(t.getMatrix())