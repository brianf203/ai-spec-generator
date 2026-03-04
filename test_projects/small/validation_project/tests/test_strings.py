import unittest
from strings import is_non_empty, has_min_length, has_max_length, is_alphanumeric, is_numeric_str
class TestStr(unittest.TestCase):
    def test_non_empty(self): self.assertTrue(is_non_empty("x"))
    def test_min_length(self): self.assertTrue(has_min_length("hello", 3))
    def test_max_length(self): self.assertTrue(has_max_length("hi", 5))
    def test_alphanumeric(self): self.assertTrue(is_alphanumeric("abc123"))
    def test_numeric_str(self): self.assertTrue(is_numeric_str("123"))