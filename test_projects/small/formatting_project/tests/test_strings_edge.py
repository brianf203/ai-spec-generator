import unittest
from strings import pad_string, truncate_string, format_number, pad_left, center_string
class TestStrEdge(unittest.TestCase):
    def test_pad_exact(self): self.assertEqual(pad_string("abc", 3), "abc")
    def test_truncate_short(self): self.assertEqual(truncate_string("hi", 5), "hi")
    def test_format_decimals(self): self.assertEqual(format_number(1.5, 1), "1.5")
    def test_center_odd(self): self.assertEqual(center_string("x", 3), " x ")