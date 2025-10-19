import mikideeplib as mk
from numpy.typing import NDArray

class Tensor():
    def __init__(self, n_input, n_output, device="CPU", array=None):
        if array is None:
            self.tensor = mk.TensorFloat(n_input, n_output, device)
        else:
            self.tensor = mk.TensorFloat(n_input, n_output, device, array)
        self.n_input = n_input
        self.n_output = n_output
        self.device = device

    def __repr__(self):
        return f"Tensor(, {self.tensor.getRows(), self.tensor.getCols()}]"
    
    def T(self):
        return Tensor(self.n_input, self.n_output, self.device, self.tensor.transpose().getMatrix())
    
    def __add__(self, other):
        return Tensor(self.n_input, self.n_output, self.tensor + other.tensor, self.device)
