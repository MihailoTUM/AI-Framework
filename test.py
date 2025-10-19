import sys
sys.path.append(r"C:\Users\drini\OneDrive\Desktop\cuda\CUDA")
import mikideeplib

t = mikideeplib.TensorFloat(3, 3, "CPU")
t.print()
print(t.getRows())
print(t.getCols())