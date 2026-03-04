import unittest
from factorial import factorial, factorial_iter, double_factorial, falling_factorial
class TestEdge(unittest.TestCase):
    def test_factorial_zero(self): self.assertEqual(factorial(0), 1)
    def test_factorial_one(self): self.assertEqual(factorial(1), 1)
    def test_double_one(self): self.assertEqual(double_factorial(1), 1)
    def test_falling_small(self): self.assertEqual(falling_factorial(5, 1), 5)