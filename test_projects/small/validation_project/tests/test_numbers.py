import unittest
from number_validation import is_positive, is_negative, is_zero, is_integer, is_in_range
class TestNum(unittest.TestCase):
    def test_pos(self): self.assertTrue(is_positive(1))
    def test_neg(self): self.assertTrue(is_negative(-1))
    def test_zero(self): self.assertTrue(is_zero(0))
    def test_integer(self): self.assertTrue(is_integer(5))
    def test_in_range(self): self.assertTrue(is_in_range(5, 0, 10))