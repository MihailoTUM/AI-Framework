import sys
sys.path.append(r"D:\AI-Framework\AI-Framework")
import AIKI
import unittest

input = AIKI.GPUTensor(2, 8)
input.print();

weights = AIKI.GPUTensor(8, 4)
bias = AIKI.GPUTensor(2, 4)

result = input @ weights + bias

relu = AIKI.Activation("RELU")

result.copyToHost()
result.print()

out = relu.forward(result)

out.copyToHost()
out.print()