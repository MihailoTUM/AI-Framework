# Beta Version (CPU-only for now)

The idea is very simple:
    A C++ Tensor class that lets python execute computationl intensive tasks such as
    matrix-multiplication on your Nvidia GPU.

You can use the Tensor library in python for your own machine learning algorithms, libraries, or frameworks. 

## Tensor (C++ side)
### Properties
- T* matrix (pointer of the matrix)
- int rows (describe the dimension of the matrix)
- int cols (describe the dimension of the matrix)
- std::string device (captures if computation was executed on the "CPU" or "CUDA")
- py:array_t<T> py_array_ref (reference to the numpy array)

### Init



## Tensor (python side)
### Properties
- rows 
- cols
- device: Optional
- numpy array (python lists are not allowed): Optional

### Init
```
import mikideeplib as mk

tensor = mk.TensorFloat(3, 3, "CPU")
```
