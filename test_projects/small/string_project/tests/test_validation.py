import unittest
from validation import is_empty, has_digit, has_alpha, starts_with, ends_with
class TestVal(unittest.TestCase):
    def test_empty(self): self.assertTrue(is_empty(""))
    def test_has_digit(self): self.assertTrue(has_digit("a1"))
    def test_has_alpha(self): self.assertTrue(has_alpha("1a"))
    def test_starts(self): self.assertTrue(starts_with("hello", "hel"))
    def test_ends(self): self.assertTrue(ends_with("hello", "lo"))