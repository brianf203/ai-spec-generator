import unittest
from factorial import factorial, factorial_iter, double_factorial, falling_factorial, rising_factorial
class Test(unittest.TestCase):
    def test_factorial(self): self.assertEqual(factorial(5), 120)
    def test_iter(self): self.assertEqual(factorial_iter(5), 120)
    def test_double(self): self.assertEqual(double_factorial(6), 48)
    def test_falling(self): self.assertEqual(falling_factorial(5, 3), 60)
    def test_rising(self): self.assertEqual(rising_factorial(2, 3), 24)