import unittest
from ops import bit_and, bit_or, bit_xor, left_shift, right_shift, count_ones
class TestOps(unittest.TestCase):
    def test_and(self): self.assertEqual(bit_and(5, 3), 1)
    def test_or(self): self.assertEqual(bit_or(5, 3), 7)
    def test_xor(self): self.assertEqual(bit_xor(5, 3), 6)
    def test_left(self): self.assertEqual(left_shift(1, 3), 8)
    def test_right(self): self.assertEqual(right_shift(8, 2), 2)
    def test_count_ones(self): self.assertEqual(count_ones(7), 3)