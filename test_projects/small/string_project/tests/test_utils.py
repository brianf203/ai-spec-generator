import unittest
from utils import count_chars, strip_whitespace, remove_spaces, first_char, last_char
class TestUtils(unittest.TestCase):
    def test_count(self): self.assertEqual(count_chars("hi"), 2)
    def test_strip(self): self.assertEqual(strip_whitespace("  x  "), "x")
    def test_remove_spaces(self): self.assertEqual(remove_spaces("a b"), "ab")
    def test_first_char(self): self.assertEqual(first_char("hi"), "h")
    def test_last_char(self): self.assertEqual(last_char("hi"), "i")