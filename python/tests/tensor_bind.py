import sys
sys.path.append(r"C:\AI-Framework\AI-Framework\build\Release")
import my_module
import unittest

t = my_module.Tensor(3, 3, 'G', True)
t.print()