import unittest
from parity import is_even, is_odd, is_divisible, next_even, prev_odd
class TestParity(unittest.TestCase):
    def test_even(self): self.assertTrue(is_even(4))
    def test_odd(self): self.assertTrue(is_odd(3))
    def test_divisible(self): self.assertTrue(is_divisible(10, 2))
    def test_next_even(self): self.assertEqual(next_even(3), 4)
    def test_prev_odd(self): self.assertEqual(prev_odd(6), 5)