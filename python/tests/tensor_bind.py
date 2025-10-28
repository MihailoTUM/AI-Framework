import sys
sys.path.append(r"D:\cuda\CUDA\build\Release")
import my_module
import unittest

class TestMyModule(unittest.TestCase):
    def test_add(self):
       tensor1 = my_module.Tensor(2, 2, "G", True)
       tensor2 = my_module.Tensor(2, 2, "G", True)
       result = tensor1 + tensor2
       result.print()

if __name__ == "__main__":
    # unittest.main(verbosity=2)
    pass

tensor1 = my_module.Tensor(2, 2, "G", True)
tensor1.print()
tensor2 = my_module.Tensor(2, 2, "G", True)
tensor2.print()
result = tensor1 + tensor2
print(result)