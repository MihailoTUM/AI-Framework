import sys
sys.path.append(r"C:\Users\drini\OneDrive\Desktop\cuda\CUDA\bin")
import mikideeplib as mk
import numpy as np

WEIGHTS = mk.TensorFloat(8, 2, "CPU")
BIAS = mk.TensorFloat(1, 2, "CPU")
activation = mk.Activation("RELU")

def forward(input):
    return activation.forward(input @ WEIGHTS + BIAS)

input_array = np.random.rand(1, 8).astype(np.float32)
INPUT = mk.TensorFloat(1, 8, "CPU", input_array)

output = forward(INPUT)
print(output.getMatrix())