import unittest
from parity import is_even, is_odd, is_divisible, next_even, prev_odd
class TestParityEdge(unittest.TestCase):
    def test_even_zero(self): self.assertTrue(is_even(0))
    def test_odd_one(self): self.assertTrue(is_odd(1))
    def test_divisible_zero(self): self.assertFalse(is_divisible(5, 0))
    def test_next_even_even(self): self.assertEqual(next_even(4), 4)
    def test_prev_odd_odd(self): self.assertEqual(prev_odd(5), 5)