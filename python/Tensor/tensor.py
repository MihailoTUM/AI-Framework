import sys
sys.path.append(r"D:\AI-Framework\AI-Framework")
import AIKI
from typing import Optional

class Tensor():
    def __init__(
        self,
        n_input: int,
        n_output: int,
        data = None
    ):
        if data is None:
            self.data = AIKI.GPUTensor(n_input, n_output)
        else:
            self.data = data

        self.n_input = n_input
        self.n_output = n_output

        self.parents = set()
        self.grads = None
        self._backward = lambda x:x

    def __repr__(self):
        return f"Tensor()"
    
    def print(self):
        self.data.print()

    def __add__(self, other):
        add = Tensor(self.n_input, self.n_output, data=(self.data + other.data))
        add.parents.add(self)
        add.parents.add(other)

        def add_backward():
            pass

        add._backward = add_backward
        return add