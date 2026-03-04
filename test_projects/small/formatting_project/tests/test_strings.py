import unittest
from strings import pad_string, truncate_string, format_number, pad_left, center_string
class TestStr(unittest.TestCase):
    def test_pad(self): self.assertEqual(pad_string("x", 3), "x  ")
    def test_truncate(self): self.assertEqual(truncate_string("hello", 3), "hel")
    def test_format(self): self.assertEqual(format_number(1.5), "1.50")
    def test_pad_left(self): self.assertEqual(pad_left("x", 3), "  x")
    def test_center(self): self.assertEqual(center_string("x", 5), "  x  ")