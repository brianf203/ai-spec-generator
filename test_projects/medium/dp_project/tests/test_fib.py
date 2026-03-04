import unittest
from fib import fib_dp, fib_dp_space, fib_dp_mod
class Test(unittest.TestCase):
    def test_fib(self): self.assertEqual(fib_dp(10), 55)
    def test_space(self): self.assertEqual(fib_dp_space(10), 55)
    def test_mod(self): self.assertEqual(fib_dp_mod(10), 55)