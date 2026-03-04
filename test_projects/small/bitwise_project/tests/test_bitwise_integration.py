import unittest
from ops import bit_and, count_ones
from mask import set_bit, get_bit
class TestIntegration(unittest.TestCase):
    def test_set_then_get(self): n=set_bit(0, 5); self.assertEqual(get_bit(n, 5), 1)
    def test_count_set_bits(self): self.assertEqual(count_ones(bit_and(15, 7)), 3)