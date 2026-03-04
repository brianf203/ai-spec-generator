import unittest
from ops import bit_or, left_shift
from mask import mask_lower
class TestSuite(unittest.TestCase):
    def test_or_shift(self): self.assertEqual(bit_or(1, 2), 3)
    def test_mask_shift(self): self.assertEqual(mask_lower(31, 4), 15)