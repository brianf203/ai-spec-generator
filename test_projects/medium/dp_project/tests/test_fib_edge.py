import unittest
from fib import fib_dp, fib_dp_space, fib_dp_mod
class TestEdge(unittest.TestCase):
    def test_fib_zero(self): self.assertEqual(fib_dp(0), 0)
    def test_fib_one(self): self.assertEqual(fib_dp(1), 1)
    def test_space_zero(self): self.assertEqual(fib_dp_space(0), 0)
    def test_mod(self): self.assertEqual(fib_dp_mod(5) % (10**9+7), fib_dp(5) % (10**9+7))