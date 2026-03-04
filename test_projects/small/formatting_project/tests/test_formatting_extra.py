"""Additional formatting edge tests."""
import unittest
from strings import pad_string, truncate_string, format_number, pad_left, center_string


class TestFormattingExtra(unittest.TestCase):
    def test_pad_string_exact_width(self):
        self.assertEqual(pad_string("abc", 3), "abc")

    def test_pad_string_custom_char(self):
        self.assertEqual(pad_string("x", 3, "0"), "x00")

    def test_truncate_exact(self):
        self.assertEqual(truncate_string("hi", 2), "hi")

    def test_format_number_zero_decimals(self):
        self.assertEqual(format_number(3.14159, 0), "3")

    def test_pad_left_custom_char(self):
        self.assertEqual(pad_left("x", 3, "-"), "--x")

    def test_center_string_odd_width(self):
        self.assertEqual(center_string("x", 5), "  x  ")
