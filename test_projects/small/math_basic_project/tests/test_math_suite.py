import unittest
from aggregate import max_list
from rounding import floor_val, ceil_val
class TestSuite(unittest.TestCase):
    def test_max(self): self.assertEqual(max_list([1, 5, 3]), 5)
    def test_floor_ceil(self): self.assertEqual(floor_val(2.7), 2); self.assertEqual(ceil_val(2.1), 3)