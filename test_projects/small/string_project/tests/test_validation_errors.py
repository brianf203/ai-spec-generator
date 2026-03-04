import unittest
from validation import is_empty, has_digit, has_alpha, starts_with, ends_with
class TestValErrors(unittest.TestCase):
    def test_empty_false(self): self.assertFalse(is_empty("x"))
    def test_no_digit(self): self.assertFalse(has_digit("abc"))
    def test_no_alpha(self): self.assertFalse(has_alpha("123"))
    def test_starts_false(self): self.assertFalse(starts_with("hi", "x"))
    def test_ends_false(self): self.assertFalse(ends_with("hi", "x"))