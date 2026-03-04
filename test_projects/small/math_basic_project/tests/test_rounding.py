import unittest
from rounding import round_to_int, round_to_n, floor_val, ceil_val
class TestRounding(unittest.TestCase):
    def test_round_int(self): self.assertEqual(round_to_int(3.7), 4)
    def test_round_n(self): self.assertEqual(round_to_n(3.14159, 2), 3.14)
    def test_floor(self): self.assertEqual(floor_val(3.7), 3)
    def test_ceil(self): self.assertEqual(ceil_val(3.2), 4)