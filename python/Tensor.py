import sys
sys.path.append(r"C:\Users\drini\OneDrive\Desktop\cuda\CUDA\bin")
import mikideeplib as mk
from numpy.typing import NDArray
from typing import Optional
import numpy as np

class Tensor():
    def __init__(
        self, 
        n_input,
        n_output, 
        device:str = "CPU", 
        data:Optional[NDArray | None] = None
    ):
        if data is None:
            self.data = mk.TensorFloat(n_input, n_output, device)
        else:
            self.data = mk.TensorFloat(n_input, n_output, device, data.getMatrix())
        self.n_input = n_input
        self.n_output = n_output
        self.device = device
        self.parents = set()
        self.grads = None
        self._backward = lambda x:x
        self.operation = ""

    def create_zero_tensor(input, output, device):
        return mk.TensorFloat(input, output, device, np.zeros(shape=(input, output)).astype(np.float32))

    def create_ones_tensor(input, output, device):
        return mk.TensorFloat(input, output, device, np.ones(shape=(input, output)).astype(np.float32))

    def __repr__(self):
        return f"Tensor(data={self.data.getMatrix()}, {self.n_input, self.n_output})"
    
    def __add__(self, other):
        add = Tensor(self.n_input, self.n_output, self.device, data=(self.data + other.data))
        add.parents.add(self)
        add.parents.add(other)
        add.operation = "+"

        def add_backward():
            if self.grads is None:
                self.grads = self.create_zero_tensor(self.n_input, self.n_output, self.device)

            parent_1 = self.parents[0]
            parent_2 = self.parents[1]
            
            if parent_1 is None:
                parent_1.grads = self.create_zero_tensor(parent_1.n_input, parent_1.n_output, parent_1.device)
            if parent_2.grads is None:
                parent_2.grads = self.create_zero_tensor(parent_2.n_input, parent_2.n_output, parent_2.device)

            parent_1.grads += self.create_ones_tensor(parent_1.n_input, parent_1.n_input, parent_1.device)
            parent_2.grads += self.create_ones_tensor(parent_2.n_input, parent_2.n_input, parent_2.device)

        add._backward = add_backward
        return add
    
    def __mul__(self, scalar):
        return Tensor(self.n_input, self.n_output, self.device, data=(self.data * scalar))
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def __matmul__(self, other):
        matmul = Tensor(self.n_input, other.n_output, self.device, data=(self.data @ other.data))
        matmul.parents.add(self)
        matmul.parents.add(other)
        matmul.operation = "@"

        def matmul_backward():
            if self.grads is None:
                self.grads = mk.TensorFloat(self.n_input, self.n_output, self.device, data=(mk.TensorFloat(self.n_input, self.n_output, np.zeros(shape=(self.n_input, self.n_output)).astype(np.float32))))

            parent_1 = self.parents[0]
            parent_2 = self.parents[1]
            
            if self.parents[0].grads is None:
                self.parents[0].grads = mk.TensorFloat(parent_1.n_input, parent_1.n_output, self.device, data=(mk.TensorFloat(parent_1.n_input, parent_1.n_output, np.zeros(shape=(parent_1.n_input, parent_1.n_output)).astype(np.float32))))

            if self.parents[1].grads is None:
                self.parents[1].grads = mk.TensorFloat(parent_2.n_input, parent_2.n_output, self.device, data=(mk.TensorFloat(parent_2.n_input, parent_2.n_output, np.zeros(shape=(parent_2.n_input, parent_2.n_output)).astype(np.float32))))

        matmul._backward = matmul_backward
        return matmul
    
    def __neg__(self):
        return Tensor(self.n_input, self.n_output, self.device, data=(-self.data))
    
    def backward(self):
        visited = set()
        graph = []
        
        def create_graph(tensor):
            for parent in tensor.parents:
                if parent not in visited:
                    visited.add(parent)
                    create_graph(parent)
            graph.append(tensor)

        create_graph(self)
        graph.reverse()

        return graph